import pytest
import torch
from helpers.utils import available_gpus, init_distributed, rerun_if_address_is_in_use
from nanotron import distributed as dist
from nanotron.config.parallelism_config import ParallelismArgs


def _check_groups_dp4_ep2(parallel_context):
    rank = dist.get_rank(parallel_context.world_pg)

    # dp_pg spans the whole DP world (size 4).
    assert parallel_context.dp_pg.size() == 4
    dp_ranks = set(dist.get_process_group_ranks(parallel_context.dp_pg))
    assert dp_ranks == {0, 1, 2, 3}

    # ep_pg is a contiguous size-2 sub-range. Expected groups: {0,1} and {2,3}.
    ep_ranks = set(dist.get_process_group_ranks(parallel_context.ep_pg))
    expected_ep_group = {0, 1} if rank in {0, 1} else {2, 3}
    assert parallel_context.ep_pg.size() == 2
    assert ep_ranks == expected_ep_group

    # expert_dp_pg is the orthogonal complement. Expected groups: {0,2} and {1,3}.
    edp_ranks = set(dist.get_process_group_ranks(parallel_context.expert_dp_pg))
    expected_edp_group = {0, 2} if rank in {0, 2} else {1, 3}
    assert parallel_context.expert_dp_pg.size() == 2
    assert edp_ranks == expected_edp_group


@pytest.mark.skipif(available_gpus() < 4, reason="needs 4 gpus")
@rerun_if_address_is_in_use()
def test_ep_subset_of_dp_dp4_ep2():
    init_distributed(tp=1, dp=4, pp=1, ep=2)(_check_groups_dp4_ep2)()


def _check_groups_dp4_ep4(parallel_context):
    # EP == DP_total -> ep_pg is the whole DP, expert_dp_pg has size 1.
    assert parallel_context.dp_pg.size() == 4
    assert parallel_context.ep_pg.size() == 4
    assert parallel_context.expert_dp_pg.size() == 1


@pytest.mark.skipif(available_gpus() < 4, reason="needs 4 gpus")
@rerun_if_address_is_in_use()
def test_ep_equals_dp_total():
    init_distributed(tp=1, dp=4, pp=1, ep=4)(_check_groups_dp4_ep4)()


def _check_world_size_no_ep_axis(parallel_context):
    # World size = TP * PP * DP, NOT * EP.
    assert parallel_context.world_pg.size() == 4


@pytest.mark.skipif(available_gpus() < 4, reason="needs 4 gpus")
@rerun_if_address_is_in_use()
def test_world_size_excludes_ep_axis():
    init_distributed(tp=1, dp=4, pp=1, ep=2)(_check_world_size_no_ep_axis)()


def _check_ep1_collapses_to_pure_dp(parallel_context):
    # EP=1: ep_pg is each rank alone (size 1); expert_dp_pg is the full DP.
    assert parallel_context.ep_pg.size() == 1
    assert parallel_context.expert_dp_pg.size() == parallel_context.dp_pg.size()


@pytest.mark.skipif(available_gpus() < 4, reason="needs 4 gpus")
@rerun_if_address_is_in_use()
def test_ep1_collapses_to_pure_dp():
    init_distributed(tp=1, dp=4, pp=1, ep=1)(_check_ep1_collapses_to_pure_dp)()


def _check_expert_mp_pg_dp4_ep2(parallel_context):
    rank = dist.get_rank(parallel_context.world_pg)
    # TP=PP=1 means expert_mp_pg should equal ep_pg (TP*PP*EP = 1*1*2 = 2).
    # Two groups: {0,1} and {2,3}.
    assert parallel_context.expert_mp_pg is not None, f"rank {rank} has no expert_mp_pg"
    members = set(dist.get_process_group_ranks(parallel_context.expert_mp_pg))
    expected = {0, 1} if rank in {0, 1} else {2, 3}
    assert members == expected, f"rank {rank}: expected {expected}, got {members}"


@pytest.mark.skipif(available_gpus() < 4, reason="needs 4 gpus")
@rerun_if_address_is_in_use()
def test_expert_mp_pg_built_for_all_ranks_dp4_ep2():
    init_distributed(tp=1, dp=4, pp=1, ep=2)(_check_expert_mp_pg_dp4_ep2)()


def _check_rank_helpers(parallel_context):
    # ep_rank_of and expert_dp_rank_of math.
    assert parallel_context.ep_rank_of(0) == 0
    assert parallel_context.ep_rank_of(1) == 1
    assert parallel_context.ep_rank_of(2) == 0
    assert parallel_context.ep_rank_of(3) == 1
    assert parallel_context.expert_dp_rank_of(0) == 0
    assert parallel_context.expert_dp_rank_of(1) == 0
    assert parallel_context.expert_dp_rank_of(2) == 1
    assert parallel_context.expert_dp_rank_of(3) == 1


@pytest.mark.skipif(available_gpus() < 4, reason="needs 4 gpus")
@rerun_if_address_is_in_use()
def test_rank_helpers_dp4_ep2():
    init_distributed(tp=1, dp=4, pp=1, ep=2)(_check_rank_helpers)()


def _check_expert_grad_reduced_on_expert_dp_pg(parallel_context):
    """When expert_dp_pg.size() > 1, expert grads should be averaged across that group."""
    if parallel_context.expert_dp_pg.size() == 1:
        return  # nothing to test in this layout
    # Build a fake expert-marked parameter and verify averaging.
    from nanotron.parallel.data_parallel.utils import mark_expert, sync_expert_gradients

    p = torch.nn.Parameter(torch.zeros(2, device="cuda"))
    mark_expert(p)
    # Seed each rank's gradient with its own world rank so the post-reduce
    # value equals the mean of world ranks within this expert_dp_pg group.
    p.grad = torch.full_like(p.data, float(dist.get_rank(parallel_context.world_pg)))

    class Holder(torch.nn.Module):
        def __init__(self, p):
            super().__init__()
            self.p = p

    holder = Holder(p)
    sync_expert_gradients(
        module=holder,
        expert_dp_pg=parallel_context.expert_dp_pg,
        reduce_op=dist.ReduceOp.AVG,
        grad_accumulator=None,
    )
    expected = sum(dist.get_process_group_ranks(parallel_context.expert_dp_pg)) / parallel_context.expert_dp_pg.size()
    assert torch.allclose(p.grad, torch.full_like(p.grad, expected))


@pytest.mark.skipif(available_gpus() < 4, reason="needs 4 gpus")
@rerun_if_address_is_in_use()
def test_expert_grad_reduced_dp4_ep2():
    init_distributed(tp=1, dp=4, pp=1, ep=2)(_check_expert_grad_reduced_on_expert_dp_pg)()


def test_parallelism_args_rejects_indivisible_ep():
    with pytest.raises(AssertionError, match="must be divisible by expert_parallel_size"):
        ParallelismArgs(dp=3, pp=1, tp=1, expert_parallel_size=2)


def test_parallelism_args_accepts_divisible_ep():
    args = ParallelismArgs(dp=4, pp=1, tp=1, expert_parallel_size=2)
    assert args.expert_parallel_size == 2


def test_is_expert_sharded_returns_false_under_ep1(monkeypatch):
    """Under EP=1 expert_pg is a singleton per rank; every TP-sharded non-expert param
    would otherwise be mismarked as expert because my_rank ∈ tp_group. Guard fixes that.
    """
    from types import SimpleNamespace

    from nanotron.parallel import parameters as _params
    from nanotron.parallel.parameters import ShardedInfo, SlicesPair

    # Mock: my_rank is 0, ep_pg is the singleton {0}. tp_pg is {0, 1} (TP=2).
    fake_pc = SimpleNamespace(expert_parallel_size=1, expert_pg=object(), tp_pg=object())

    def fake_get_global_ranks(group):
        # ep_pg singleton vs tp_pg full set -- but the guard should short-circuit anyway.
        if group is fake_pc.expert_pg:
            return (0,)
        return (0, 1)

    monkeypatch.setattr(_params.dist, "get_global_ranks", fake_get_global_ranks)

    # A TP-sharded non-expert param: its global_ranks contain my_rank (0).
    info = ShardedInfo(
        global_ranks=(0, 1),
        local_global_slices_pairs=(SlicesPair(local_slices=(slice(0, 1),), global_slices=(slice(0, 2),)),),
        unsharded_shape=(2,),
    )
    # Without the EP=1 guard, this would return True (singleton {0} ⊆ {0, 1}).
    assert info.is_expert_sharded(fake_pc) is False
    # And TP-sharded check still works as expected.
    assert info.is_tp_sharded(fake_pc) is True


def test_is_expert_sharded_still_works_under_ep_gt_1(monkeypatch):
    """When EP > 1, is_expert_sharded should fall through to the normal subset test."""
    from types import SimpleNamespace

    from nanotron.parallel import parameters as _params
    from nanotron.parallel.parameters import ShardedInfo, SlicesPair

    fake_pc = SimpleNamespace(expert_parallel_size=2, expert_pg=object())

    def fake_get_global_ranks(group):
        # expert_pg of size 2, e.g. {0, 1}.
        return (0, 1)

    monkeypatch.setattr(_params.dist, "get_global_ranks", fake_get_global_ranks)

    # Param whose global_ranks are a superset of expert_pg ranks -> expert.
    expert_info = ShardedInfo(
        global_ranks=(0, 1, 2, 3),
        local_global_slices_pairs=(SlicesPair(local_slices=(slice(0, 1),), global_slices=(slice(0, 4),)),),
        unsharded_shape=(4,),
    )
    assert expert_info.is_expert_sharded(fake_pc) is True

    # Param whose global_ranks do NOT contain all of expert_pg -> not expert.
    non_expert_info = ShardedInfo(
        global_ranks=(0,),
        local_global_slices_pairs=(SlicesPair(local_slices=(slice(0, 1),), global_slices=(slice(0, 1),)),),
        unsharded_shape=(1,),
    )
    assert non_expert_info.is_expert_sharded(fake_pc) is False


def test_sync_gradients_across_dp_skips_experts(monkeypatch):
    """sync_gradients_across_dp must NOT reduce expert-marked params -- those are reduced
    on expert_dp_pg by sync_expert_gradients, and a second reduce on dp_pg would mix in
    non-replica peers.
    """
    import torch

    from nanotron.parallel.data_parallel import utils as _du
    from nanotron.parallel.data_parallel.utils import mark_expert, sync_gradients_across_dp

    # Fake DP process group; only its size matters for our recorder.
    class FakePG:
        def __init__(self, size):
            self._size = size

        def size(self):
            return self._size

    reduced = []

    def fake_all_reduce(tensor, op=None, group=None):
        # Record which tensor was reduced and average it by group size (simulating AVG).
        reduced.append(id(tensor))
        tensor.div_(group.size())

    monkeypatch.setattr(_du.dist, "all_reduce", fake_all_reduce)

    expert_param = torch.nn.Parameter(torch.zeros(2))
    mark_expert(expert_param)
    expert_param.grad = torch.full((2,), 7.0)
    non_expert_param = torch.nn.Parameter(torch.zeros(2))
    non_expert_param.grad = torch.full((2,), 4.0)

    class Holder(torch.nn.Module):
        def __init__(self, ep, nep):
            super().__init__()
            self.ep = ep
            self.nep = nep

    holder = Holder(expert_param, non_expert_param)
    dp_pg = FakePG(size=2)

    import nanotron.distributed as _dist

    sync_gradients_across_dp(
        module=holder,
        dp_pg=dp_pg,
        reduce_op=_dist.ReduceOp.AVG,
        grad_accumulator=None,
    )

    # Expert grad unchanged.
    assert torch.allclose(expert_param.grad, torch.full((2,), 7.0))
    # Non-expert grad averaged (divided by group size in our fake reduce).
    assert torch.allclose(non_expert_param.grad, torch.full((2,), 2.0))
    # Only the non-expert param was passed to all_reduce.
    assert id(non_expert_param.grad) in reduced
    assert id(expert_param.grad) not in reduced
