# src/nanotron/training_manager_nanotron.py
"""
NanotronULFMTrainingManager: Adapts ULFMTrainingManager for nanotron's
pipeline-engine-based training loop.

Gradient sync strategy: DEFERRED PER-BUCKET ALLREDUCE via DDP comm hook.
  - The pipeline engine uses no_sync() on early microbatches and
    ddp_trigger_sync_in_bwd() on the last microbatch.
  - On the last microbatch, DDP hooks fire per-bucket. Our deferred hook
    snapshots each bucket and returns a pre-resolved Future (no blocking).
  - After the pipeline completes, fire_deferred_allreduces() triggers the
    actual ULFM allreduces per-bucket via the orchestrator.
  - Per-bucket failure handling through the orchestrator's handle_work_completion.
"""

import logging

import torch
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

        self.ulfm_manager.prepare_iteration(...)
        outputs = self.pipeline_engine.train_batch_iter(...)
        # ... mp_pg barrier + dp_pg consensus ...
        self.ulfm_manager.fire_deferred_allreduces()   # post-pipeline grad sync
        stepped = self.ulfm_manager.after_pipeline(optimizer)
        if not stepped:
            return [], None
    """

    def __init__(
        self,
        ddp_model: DistributedDataParallel,
        dp_pg: ULFM.ProcessGroupULFM,
        grad_accum_steps: int = 1,
        policy_type: str = "static",
        **policy_kwargs,
    ):
        if not isinstance(dp_pg, ULFM.ProcessGroupULFM):
            raise ValueError(
                f"dp_pg must be a ProcessGroupULFM, got {type(dp_pg)}. "
                "Ensure you are using ULFMParallelContext."
            )

        self.failure_strategy = "continue"
        self.process_group = dp_pg

        policy = create_policy(
            policy_type=policy_type,
            initial_grad_accum_steps=grad_accum_steps,
            enable_auto_repair=True,
            **policy_kwargs,
        )

        ulfm_opts = self._create_ulfm_opts(policy_type)

        rank = dist.get_rank()
        self.txn = StepTxnOrchestrator(
            rank=rank, pg=dp_pg, policy=policy, ulfm_opts=ulfm_opts
        )

        self.ddp_model = ddp_model
        self._micro_in_window = 0

        # Register deferred comm hook — snapshots + queues per-bucket,
        # returns pre-resolved Futures so finalize_backward never blocks.
        self._register_deferred_hook(ulfm_opts)

        logger.debug(
            f"[Rank {rank}] NanotronULFMTrainingManager initialized "
            f"(deferred hook registered): "
            f"policy={policy_type}, grad_accum={grad_accum_steps}"
        )

    def _register_deferred_hook(self, ulfm_opts):
        from ulfm_collectives.ulfm_hook import create_ulfm_deferred_hook, HookState

        hstate = HookState(pg=self.process_group, orchestrator=self.txn)
        hook = create_ulfm_deferred_hook(ulfm_opts=ulfm_opts)
        self.ddp_model.register_comm_hook(state=hstate, hook=hook)

    # ------------------------------------------------------------------
    # Nanotron lifecycle interface
    # ------------------------------------------------------------------

    def prepare_iteration(self, iteration_step: int, n_microbatches: int) -> None:
        self.txn.update_progress(
            microbatch_idx=0,
            total_microbatches=n_microbatches,
            macrobatch_idx=iteration_step,
        )
        self._notify_window_start()
        self._on_consensus_step()
        self._on_grad_sync_step()

    def fire_deferred_allreduces(self):
        """
        Post-pipeline per-bucket ULFM gradient allreduce.

        Delegates to the orchestrator which fires the actual ULFM allreduces
        for each bucket that was queued by the deferred hook.
        """
        self.txn.fire_deferred_allreduces()

    def after_pipeline(self, optimizer) -> bool:
        """
        Call after fire_deferred_allreduces(). Handles restore + optimizer step.
        """
        n_microbatches = self._get_grad_accum_steps()
        state = self._on_microbatch_complete(n_microbatches - 1)
        restore_mode = self._get_restore_mode()

        if not state.at_iteration_boundary:
            logger.warning(
                f"[Rank {self.txn._rank}] after_pipeline: not at iteration boundary. "
                "Skipping optimizer step."
            )
            optimizer.zero_grad()
            self.txn.mark_iteration_end()
            self._micro_in_window = 0
            return False

        if restore_mode == GradRestoreMode.NON_BLOCKING:
            logger.warning(
                f"[Rank {self.txn._rank}] NON_BLOCKING restore at policy boundary — "
                "skipping optimizer step."
            )
            optimizer.zero_grad()
            self.txn.mark_iteration_end()
            self._micro_in_window = 0
            return False

        if restore_mode == GradRestoreMode.BLOCKING:
            logger.info(
                f"[Rank {self.txn._rank}] BLOCKING gradient restore before optimizer step."
            )
            self._start_restore_gradients_blocking()

        # Normalize: allreduce uses SUM, divide by effective_batch_size
        div_factor = self._get_grad_div_factor()
        for p in self.ddp_model.parameters():
            if p.grad is not None:
                p.grad.div_(div_factor)

        optimizer.step()
        optimizer.zero_grad()

        self._on_step_committed()
        self._micro_in_window = 0

        return True
