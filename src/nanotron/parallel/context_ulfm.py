# src/nanotron/parallel/context_ulfm.py
"""
ULFMParallelContext: ParallelContext variant that initializes ULFM as the world
backend and pins TP/PP/EP/MP subgroups to explicit NCCL.

The DP process group inherits the ULFM default backend so the ULFM comm hook
can intercept gradient allreduces for fault-tolerant DP training.
"""

import os
from typing import Optional

import numpy as np

import nanotron.distributed as dist
from nanotron.distributed import initialize_torch_distributed_ulfm
from nanotron.parallel.context import ParallelContext


class ULFMParallelContext(ParallelContext):
    """
    Parallel context that uses ULFM for the world/DP groups and NCCL for TP/PP/EP.

    Initialization order:
      1. Call initialize_torch_distributed_ulfm() so dist.is_initialized() is True
         before the parent __init__ can call the NCCL initializer.
      2. Call super().__init__(..., backend="ulfm") which creates world_pg as ULFM.
      3. Override _init_parallel_groups() to pass backend="nccl" explicitly to all
         groups except dp_pg, which gets no explicit backend (inherits ULFM).
    """

    def __init__(
        self,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        data_parallel_size: int,
        expert_parallel_size: int = 1,
    ):
        # Pre-initialize with ULFM so the parent's guard (if not dist.is_initialized())
        # skips the NCCL initialize_torch_distributed() call.
        if not dist.is_initialized():
            initialize_torch_distributed_ulfm()

        # Pass backend="ulfm" to bypass the assert and so world_pg uses ULFM.
        super().__init__(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=data_parallel_size,
            expert_parallel_size=expert_parallel_size,
            backend="ulfm",
        )

    def _init_parallel_groups(self):
        """
        Same layout as parent but TP/PP/EP/MP groups use explicit NCCL backend.
        dp_pg is created without a backend argument so it inherits ULFM.
        """
        dist.barrier()
        world_size = int(os.environ["WORLD_SIZE"])
        ranks = np.arange(0, world_size).reshape(
            (
                self.expert_parallel_size,
                self.pipeline_parallel_size,
                self.data_parallel_size,
                self.tensor_parallel_size,
            )
        )
        self.world_ranks_to_pg = {}

        # TP: NCCL (intra-replica, no failure expected)
        self.tp_pg = self.create_new_group(
            ranks.transpose((0, 1, 2, 3)).reshape((-1, self.tensor_parallel_size)),
            backend="nccl",
        )
        # DP: ULFM (cross-replica, needs fault tolerance) — no backend → inherits ULFM
        self.dp_pg = self.create_new_group(
            ranks.transpose((3, 0, 1, 2)).reshape((-1, self.data_parallel_size)),
            backend=None,
        )
        # PP: NCCL (intra-replica pipeline, no failure expected)
        self.pp_pg = self.create_new_group(
            ranks.transpose((2, 3, 0, 1)).reshape((-1, self.pipeline_parallel_size)),
            backend="nccl",
        )
        # Expert: NCCL
        self.expert_pg = self.create_new_group(
            ranks.transpose((1, 2, 3, 0)).reshape((-1, self.expert_parallel_size)),
            backend="nccl",
        )
        # Model parallel (TP+PP+EP for a given DP rank): NCCL
        self.mp_pg = self.create_new_group(
            [ranks[:, :, dp_rank, :].reshape(-1) for dp_rank in range(self.data_parallel_size)],
            backend="nccl",
        )
        # TP+Expert combined: NCCL
        self.tp_and_expert_pg = self.create_new_group(
            [
                ranks[:, pp_rank, dp_rank, :].reshape(-1)
                for pp_rank in range(self.pipeline_parallel_size)
                for dp_rank in range(self.data_parallel_size)
            ],
            backend="nccl",
        )
        self.world_rank_matrix: np.ndarray = ranks
