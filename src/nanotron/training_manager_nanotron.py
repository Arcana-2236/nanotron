# src/nanotron/training_manager_nanotron.py
"""
NanotronULFMTrainingManager: Adapts ULFMTrainingManager for nanotron's
pipeline-engine-based training loop.

The parent class (ULFMTrainingManager) owns the forward→backward→optimizer loop.
Nanotron's pipeline engine owns forward+backward; we bracket it with lifecycle calls.

All ULFM recovery logic (orchestrator, policy, hook, gradient restore) is inherited
unchanged from ULFMTrainingManager. This class only changes:
  - __init__: accepts an existing DDP model + explicit dp_pg instead of creating DDP
  - prepare_iteration(): called before pipeline_engine.train_batch_iter()
  - after_pipeline(): called after pipeline_engine.train_batch_iter(); handles grad
    restoration and optimizer step, mirrors the iteration-boundary block in
    ULFMTrainingManager.train_step()
"""

import logging

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import ulfm_collectives as ULFM
from ulfm_collectives.training_manager import ULFMTrainingManager
from ulfm_collectives.orchestrator import StepTxnOrchestrator
from ulfm_collectives.policy import create_policy, GradRestoreMode

logger = logging.getLogger(__name__)


class NanotronULFMTrainingManager(ULFMTrainingManager):
    """
    ULFMTrainingManager adapted for nanotron's pipeline-engine training loop.

    Usage pattern in ULFMDistributedTrainer.training_step():

        self.ulfm_manager.prepare_iteration(self.iteration_step, self.n_micro_batches_per_batch)
        outputs = self.pipeline_engine.train_batch_iter(model=self.model, ...)
        stepped = self.ulfm_manager.after_pipeline(self.optimizer)
        if not stepped:
            return [], None   # failure detected; skip logging/checkpoint
    """

    def __init__(
        self,
        ddp_model: DistributedDataParallel,
        dp_pg: ULFM.ProcessGroupULFM,
        grad_accum_steps: int = 1,
        policy_type: str = "static",
        **policy_kwargs,
    ):
        """
        Args:
            ddp_model: The model already wrapped in DistributedDataParallel by nanotron.
                       Must use dp_pg as its process_group.
            dp_pg: The ULFM DP process group (from ULFMParallelContext.dp_pg).
            grad_accum_steps: Matches nanotron's batch_accumulation_per_replica.
            policy_type: "static" (initial target) or "adaptive".
            **policy_kwargs: Forwarded to create_policy(). When policy_type="static"
                (the default), `initial_world_size` is REQUIRED, e.g.:
                    NanotronULFMTrainingManager(..., initial_world_size=dist.get_world_size())
        """
        # Intentionally do NOT call super().__init__() — the parent creates its own
        # DistributedDataParallel wrapper and binds to dist.group.WORLD.
        # We manually initialize the same components with the correct process group.

        if not isinstance(dp_pg, ULFM.ProcessGroupULFM):
            raise ValueError(
                f"dp_pg must be a ProcessGroupULFM, got {type(dp_pg)}. "
                "Ensure you are using ULFMParallelContext."
            )

        self.failure_strategy = "continue"
        self.process_group = dp_pg  # used by _register_ulfm_hook → HookState.pg

        policy = create_policy(
            policy_type=policy_type,
            initial_grad_accum_steps=grad_accum_steps,
            enable_auto_repair=True,
            **policy_kwargs,
        )

        ulfm_opts = self._create_ulfm_opts(policy_type)

        rank = dist.get_rank()  # global rank, used for logging in orchestrator
        self.txn = StepTxnOrchestrator(
            rank=rank, pg=dp_pg, policy=policy, ulfm_opts=ulfm_opts
        )

        # Use nanotron's already-created DDP model (do not wrap again)
        self.ddp_model = ddp_model

        # Register ULFM comm hook on the existing DDP model
        self._register_ulfm_hook(ulfm_opts=ulfm_opts)

        self._micro_in_window = 0  # inherited counter; kept at 0 (pipeline handles microbatches)

        logger.debug(
            f"[Rank {rank}] NanotronULFMTrainingManager initialized: "
            f"policy={policy_type}, grad_accum={grad_accum_steps}"
        )

    # ------------------------------------------------------------------
    # Nanotron lifecycle interface
    # ------------------------------------------------------------------

    def prepare_iteration(self, iteration_step: int, n_microbatches: int) -> None:
        """
        Call immediately before pipeline_engine.train_batch_iter().

        Mirrors the window-start and pre-backward setup in ULFMTrainingManager.train_step():
          - txn.update_progress(): tells orchestrator where we are
          - txn.on_iteration_start(): resets hook counter, notifies policy of window start
          - consensus(): all surviving ranks agree on any stale failures before backward
          - on_grad_sync_step_prepare(): unquiesces the communicator before backward

        Note: consensus is called before the pipeline engine (before all forwards),
        which is slightly earlier than the original (between last forward and backward).
        This is acceptable for the initial implementation: it catches stale failures
        before the allreduce, while new failures during backward are caught by the hook.
        """
        self.txn.update_progress(
            microbatch_idx=0,
            total_microbatches=n_microbatches,
            macrobatch_idx=iteration_step,
        )
        self._notify_window_start()   # txn.on_iteration_start()
        self._on_consensus_step()     # consensus() agree on stale failures
        self._on_grad_sync_step()     # txn.on_grad_sync_step_prepare() → unquiesce

    def after_pipeline(self, optimizer) -> bool:
        """
        Call immediately after pipeline_engine.train_batch_iter().

        Mirrors the iteration-boundary block in ULFMTrainingManager.train_step().
        The pipeline engine has already run all microbatch forwards and backwards;
        the ULFM hook fired during the last backward pass.

        Handles three cases from get_restore_plan():
          SKIP        → no failure; normal optimizer step
          BLOCKING    → failure detected mid-step; synchronously restore+re-reduce
                        gradients, then optimizer step
          NON_BLOCKING → failure at StaticWorldPolicy boundary (complex case);
                         skip optimizer step for this iteration (deferred support)

        Returns:
            True  if optimizer.step() was called (training made progress)
            False if the step was skipped (failure at policy boundary, or unexpected
                  non-boundary state)
        """
        # Treat the entire pipeline run as completing all grad_accum microbatches.
        n_microbatches = self._get_grad_accum_steps()
        state = self._on_microbatch_complete(n_microbatches - 1)
        restore_mode = self._get_restore_mode()

        if not state.at_iteration_boundary:
            # Unexpected: pipeline should always complete all microbatches.
            logger.warning(
                f"[Rank {self.txn._rank}] after_pipeline: not at iteration boundary "
                f"(n_microbatches={n_microbatches}, policy_accum={n_microbatches}). "
                "Skipping optimizer step."
            )
            optimizer.zero_grad(set_to_none=False)
            self.txn.mark_iteration_end()
            self._micro_in_window = 0
            return False

        if restore_mode == GradRestoreMode.NON_BLOCKING:
            # StaticWorldPolicy boundary case: policy wants to extend the grad-accum
            # window with extra microbatches. Nanotron's pipeline engine has already
            # finished, so we cannot add more microbatches this iteration.
            # Skip the optimizer step; the policy will re-adjust on the next iteration.
            # TODO: implement full StaticWorldPolicy boundary support by interleaving
            # additional pipeline engine calls.
            logger.warning(
                f"[Rank {self.txn._rank}] NON_BLOCKING restore at policy boundary — "
                "skipping optimizer step (full boundary support deferred)."
            )
            optimizer.zero_grad(set_to_none=False)
            self.txn.mark_iteration_end()
            self._micro_in_window = 0
            return False

        # BLOCKING restore: failure detected, re-sync gradients before committing.
        if restore_mode == GradRestoreMode.BLOCKING:
            logger.info(
                f"[Rank {self.txn._rank}] BLOCKING gradient restore before optimizer step."
            )
            self._start_restore_gradients_blocking()

        # Normalize gradients: ULFM hook uses ReduceOp.SUM, not AVG.
        # Divide by effective_batch_size to match AVG semantics expected by the optimizer.
        div_factor = self._get_grad_div_factor()  # txn.effective_batch_size
        for p in self.ddp_model.parameters():
            if p.grad is not None:
                p.grad.div_(div_factor)

        optimizer.step()
        optimizer.zero_grad(set_to_none=False)

        self._on_step_committed()   # txn.after_successful_commit(): advance policy, reset epoch state
        self._micro_in_window = 0

        return True
