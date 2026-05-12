import pytest
import torch
from helpers.utils import available_gpus, init_distributed, rerun_if_address_is_in_use
from nanotron import distributed as dist


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
