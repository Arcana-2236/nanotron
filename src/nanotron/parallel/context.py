import os
from typing import Literal, Tuple, Annotated

import numpy as np
import torch

import nanotron.distributed as dist

DistributedBackend = Literal["gloo", "mpi", "nccl"]


class ParallelContext:
    def __init__(
        self,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        data_parallel_size: int,
        expert_parallel_size: int = 1,
        backend: DistributedBackend = "nccl",
    ):
        """Initialize parallel context.

        Layout: world_size = TP * PP * DP. `expert_parallel_size` subdivides DP
        (it is NOT multiplied into world size). Must satisfy DP % EP == 0.
        """
        world_size = int(os.environ["WORLD_SIZE"])

        assert (
            data_parallel_size % expert_parallel_size == 0
        ), f"data_parallel_size ({data_parallel_size}) must be divisible by expert_parallel_size ({expert_parallel_size})."
        expected = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
        if expected != world_size:
            raise ValueError(
                f"TP*PP*DP ({expected}) must equal world size ({world_size}). "
                f"Note: under EP-as-subset-of-DP, EP is NOT multiplied into world size."
            )

        if not dist.is_available():
            raise ValueError("torch.distributed is not available as a package, please install it.")

        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.data_parallel_size = data_parallel_size
        self.expert_parallel_size = expert_parallel_size

        self._groups = {}

        self.set_device()

        assert backend == "nccl", "Only nccl backend is supported for now."

        if not dist.is_initialized():
            dist.initialize_torch_distributed()

        ranks = list(range(world_size))
        process_group = dist.new_group(
            ranks=ranks,
            backend=dist.get_backend(),
        )
        self.world_pg = process_group

        self._init_parallel_groups()

    def _init_parallel_groups(self):
        """Initialize 3D parallelism's all process groups under the new EP-as-subset-of-DP layout."""
        dist.barrier()
        world_size = int(os.environ["WORLD_SIZE"])
        # New 3-D layout: [PP, DP_total, TP]
        ranks = np.arange(0, world_size).reshape(
            (
                self.pipeline_parallel_size,
                self.data_parallel_size,
                self.tensor_parallel_size,
            )
        )
        self.world_ranks_to_pg = {}

        # TP: vary tp_rank with everything else fixed.
        self.tp_pg = self.create_new_group(ranks.reshape((-1, self.tensor_parallel_size)))
        # DP: vary dp_rank with everything else fixed.
        self.dp_pg = self.create_new_group(ranks.transpose((2, 0, 1)).reshape((-1, self.data_parallel_size)))
        # PP: vary pp_rank with everything else fixed.
        self.pp_pg = self.create_new_group(ranks.transpose((1, 2, 0)).reshape((-1, self.pipeline_parallel_size)))

        # EP: contiguous size-`expert_parallel_size` sub-range INSIDE dp_pg.
        # For each (pp, tp), split the dp axis into groups of size EP.
        ep = self.expert_parallel_size
        num_ep_groups_per_slot = self.data_parallel_size // ep
        ep_groups = []
        for pp_rank in range(self.pipeline_parallel_size):
            for tp_rank in range(self.tensor_parallel_size):
                for g in range(num_ep_groups_per_slot):
                    ep_groups.append(ranks[pp_rank, g * ep : (g + 1) * ep, tp_rank])
        self.ep_pg = self.create_new_group(np.array(ep_groups))

        # expert_dp: orthogonal complement of EP inside DP -- same intra-EP rank across all EP groups.
        edp_groups = []
        for pp_rank in range(self.pipeline_parallel_size):
            for tp_rank in range(self.tensor_parallel_size):
                for intra in range(ep):
                    edp_groups.append(
                        np.array([ranks[pp_rank, g * ep + intra, tp_rank] for g in range(num_ep_groups_per_slot)])
                    )
        self.expert_dp_pg = self.create_new_group(np.array(edp_groups))

        # Keep `expert_pg` as an alias for `ep_pg` so callers in moe.py and serialize/* keep working.
        self.expert_pg = self.ep_pg

        # MP (non-expert clip-grad): TP x PP for a given dp_rank.
        self.mp_pg = self.create_new_group(
            [ranks[:, dp_rank, :].reshape(-1) for dp_rank in range(self.data_parallel_size)]
        )

        # expert MP (expert clip-grad): TP x PP x EP for a given expert_dp coordinate.
        expert_mp_groups = []
        for intra in range(ep):
            for edp in range(num_ep_groups_per_slot):
                expert_mp_groups.append(
                    np.array(
                        [
                            ranks[pp_rank, edp * ep + ep_inner, tp_rank]
                            for pp_rank in range(self.pipeline_parallel_size)
                            for tp_rank in range(self.tensor_parallel_size)
                            for ep_inner in range(ep)
                        ]
                    )
                )
                break  # all `intra` produce the same expert-MP membership; one is enough
        # Deduplicate by sorted tuple (create_new_group already does this, but be explicit).
        self.expert_mp_pg = self.create_new_group(np.array(expert_mp_groups))

        # tp_and_expert_pg: TP x EP for a given (pp, expert_dp) -- used by SparseMLP marker.
        tae_groups = []
        for pp_rank in range(self.pipeline_parallel_size):
            for edp in range(num_ep_groups_per_slot):
                tae_groups.append(
                    np.array(
                        [
                            ranks[pp_rank, edp * ep + ep_inner, tp_rank]
                            for tp_rank in range(self.tensor_parallel_size)
                            for ep_inner in range(ep)
                        ]
                    )
                )
        self.tp_and_expert_pg = self.create_new_group(np.array(tae_groups))

        self.world_rank_matrix: np.ndarray = ranks

    def create_new_group(self, all_groups_ranks: np.ndarray) -> dist.ProcessGroup:
        dist.barrier()
        rank = int(os.environ["RANK"])
        new_group_containing_rank = None
        for group_ranks in all_groups_ranks:
            sorted_ranks = tuple(sorted(group_ranks))

            # add new group to `world_ranks_to_pg`
            if sorted_ranks not in self.world_ranks_to_pg:
                new_group = dist.new_group(ranks=group_ranks)
                self.world_ranks_to_pg[sorted_ranks] = new_group
            else:
                new_group = self.world_ranks_to_pg[sorted_ranks]

            if rank in sorted_ranks:
                new_group_containing_rank = new_group
        dist.barrier()
        return new_group_containing_rank

    def set_device(self):
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        # NOTE: Set the device id.
        # `torch.cuda.device_count` should return the number of device on a single node.
        # We assume the nodes to be homogeneous (same number of gpus per node)
        device_id = local_rank
        torch.cuda.set_device(torch.cuda.device(device_id))

    def get_local_ranks(self, world_rank: int) -> Tuple[int, int, int]:
        return tuple(i.item() for i in np.where(self.world_rank_matrix == world_rank))

    def destroy(self):
        if not dist.is_initialized():
            return

        dist.barrier()
        dist.destroy_process_group()

    def get_global_rank(
        self,
        pp_rank: int,
        dp_rank: int,
        tp_rank: int,
    ) -> np.int64:
        """
        Get the global rank based on the specified ranks in different parallel groups.

        Under the new EP-as-subset-of-DP layout, the world_rank_matrix is 3-D
        [PP, DP_total, TP]; the EP coordinate is embedded in dp_rank (specifically,
        ep_rank = dp_rank % expert_parallel_size).

        :param pp_rank: int, Rank in the pipeline parallel group.
        :param dp_rank: int, Rank in the data parallel group (spans the full DP world).
        :param tp_rank: int, Rank in the tensor parallel group.

        :return: numpy.int64, The global rank.
        """
        return self.world_rank_matrix[pp_rank, dp_rank, tp_rank]