import pytest
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


def test_parallelism_args_rejects_indivisible_ep():
    with pytest.raises(AssertionError, match="must be divisible by expert_parallel_size"):
        ParallelismArgs(dp=3, pp=1, tp=1, expert_parallel_size=2)


def test_parallelism_args_accepts_divisible_ep():
    args = ParallelismArgs(dp=4, pp=1, tp=1, expert_parallel_size=2)
    assert args.expert_parallel_size == 2
