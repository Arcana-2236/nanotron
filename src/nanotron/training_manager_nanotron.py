# src/nanotron/training_manager_nanotron.py
"""
NanotronULFMTrainingManager: Adapts ULFMTrainingManager for nanotron's
pipeline-engine-based training loop.

Gradient sync strategy: DEFERRED PER-BUCKET ALLREDUCE (no DDP hooks).
  - The pipeline engine always uses no_sync() so DDP's reducer never fires.
  - At init, we compute the same bucket assignment DDP would use, giving us
    per-bucket parameter groups.
  - After the pipeline completes, fire_deferred_allreduces() flattens each
    bucket's gradients into a contiguous buffer, snapshots it, allreduces
    via ULFM, and scatters the reduced values back to param.grad.
  - Per-bucket failure handling through the orchestrator's handle_work_completion.
"""

import contextlib
import logging
from typing import List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import ulfm_collectives as ULFM
from ulfm_collectives.training_manager import ULFMTrainingManager
from ulfm_collectives.orchestrator import StepTxnOrchestrator
from ulfm_collectives.policy import create_policy, GradRestoreMode
from ulfm_collectives.failure_simulator import get_failure_simulator

logger = logging.getLogger(__name__)


def _build_bucket_groups(
    ddp_model: DistributedDataParallel,
) -> List[List[torch.nn.Parameter]]:
    """
    Build parameter bucket groups matching DDP's bucket assignment.

    Returns a list of buckets, each containing the parameters in that bucket.
    """
    params = [p for p in ddp_model.parameters() if p.requires_grad]
    if not params:
        return []

    bucket_cap = ddp_model.bucket_bytes_cap
    bucket_size_limits = [bucket_cap]
    if getattr(ddp_model, "bucket_bytes_cap_default", False):
        bucket_size_limits = [dist._DEFAULT_FIRST_BUCKET_BYTES, bucket_cap]

    expect_sparse = [False] * len(params)
    bucket_indices, _ = dist._compute_bucket_assignment_by_size(
        params, bucket_size_limits, expect_sparse
    )

    buckets = []
    for indices in bucket_indices:
        buckets.append([params[i] for i in indices])
    return buckets


class NanotronULFMTrainingManager(ULFMTrainingManager):
    """
    ULFMTrainingManager adapted for nanotron's pipeline-engine training loop.

    Usage pattern in ULFMDistributedTrainer.training_step():

        self.ulfm_manager.prepare_iteration(...)
        outputs = self.pipeline_engine.train_batch_iter(...)  # always no_sync
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

        # Build bucket groups matching DDP's bucket assignment.
        self._bucket_groups = _build_bucket_groups(ddp_model)

        self._micro_in_window = 0

        logger.debug(
            f"[Rank {rank}] NanotronULFMTrainingManager initialized "
            f"({len(self._bucket_groups)} buckets): "
            f"policy={policy_type}, grad_accum={grad_accum_steps}"
        )

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

        For each bucket: flatten grads → snapshot → allreduce → scatter back.
        Per-bucket failure handling via handle_work_completion().
        """
        if not self._bucket_groups:
            return

        torch.cuda.synchronize()

        opts = torch.distributed.AllreduceOptions()
        opts.reduceOp = torch.distributed.ReduceOp.SUM
        ulfm_opts = self.txn.ulfm_opts

        for bucket_index, bucket_params in enumerate(self._bucket_groups):
            grads = [p.grad for p in bucket_params if p.grad is not None]
            if not grads:
                continue

            # Flatten into contiguous buffer
            flat = torch._utils._flatten_dense_tensors(grads)

            # Snapshot for restoration on failure
            self.txn.on_bucket_snapshot(flat, bucket_index, self.txn.dp_pg)

            # ULFM allreduce (in-place on flat buffer)
            work = self.txn.dp_pg.ulfm_allreduce([flat], opts=opts, ulfm_opts=ulfm_opts)
            work.wait()

            # Failure injection + work completion handling
            _sim = get_failure_simulator()
            ctx = (
                _sim.may_fail_here("post-allreduce")
                if _sim is not None
                else contextlib.nullcontext()
            )
            with ctx:
                self.txn.handle_work_completion(work, bucket_index=bucket_index)

            self.txn.increment_hook_counter()

            # Scatter reduced values back to param.grad
            for param, unflat in zip(
                [p for p in bucket_params if p.grad is not None],
                torch._utils._unflatten_dense_tensors(flat, grads),
            ):
                param.grad.copy_(unflat)

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
