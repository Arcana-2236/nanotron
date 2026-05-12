from contextlib import contextmanager
from typing import Optional

import torch
from nanotron import distributed as dist
from nanotron.optim.gradient_accumulator import GradientAccumulator
from torch import nn

# Single source of truth for the expert-parameter marker. Use mark_expert /
# is_expert_param accessors rather than touching the attribute directly so a
# typo can't silently degrade an expert param into a non-expert one (which
# would skip the expert_dp_pg all-reduce without raising).
EXPERT_PARAM_ATTR = "_is_expert"


def mark_expert(param: torch.nn.Parameter, value: bool = True) -> None:
    """Tag a parameter as expert (or non-expert) for downstream grad/clip routing."""
    setattr(param, EXPERT_PARAM_ATTR, value)


def is_expert_param(param: torch.nn.Parameter) -> bool:
    """Return True iff `param` was tagged via mark_expert."""
    return getattr(param, EXPERT_PARAM_ATTR, False)


@contextmanager
def ddp_trigger_sync_in_bwd(model_ddp):
    """Trigger the sync of the gradients in the next backward pass of DDP model."""
    assert isinstance(model_ddp, torch.nn.parallel.DistributedDataParallel)
    old_require_backward_grad_sync = model_ddp.require_backward_grad_sync
    old_require_forward_param_sync = model_ddp.require_forward_param_sync

    model_ddp.require_backward_grad_sync = True
    model_ddp.require_forward_param_sync = True
    # https://github.com/pytorch/pytorch/blob/master/torch/csrc/distributed/c10d/reducer.cpp#L1325-L1356
    model_ddp.reducer.prepare_for_backward([])
    try:
        yield
    finally:
        model_ddp.require_backward_grad_sync = old_require_backward_grad_sync
        model_ddp.require_forward_param_sync = old_require_forward_param_sync


def sync_gradients_across_dp(
    module: nn.Module,
    dp_pg: dist.ProcessGroup,
    reduce_op: dist.ReduceOp,
    grad_accumulator: Optional[GradientAccumulator],
    **sync_options,
):
    """Sync gradients across data parallelism.

    Args:
        module: The module to sync gradients for.
        dp_pg: The data parallelism process group.
        reduce_op: The reduce operation to use.
        grad_accumulator: The gradient accumulator to use.
        sync_options: Additional options given when using `grad_accumulator`. Please look at `GradientAccumulator.sync_gradients_across_dp` for documentation

    Note:
        Expert-marked parameters are skipped here: they are reduced on `expert_dp_pg`
        by `sync_expert_gradients` and a second reduce on `dp_pg` would mix in
        non-replica peers.
    """
    if grad_accumulator is not None:
        # This is an optimized path that
        grad_accumulator.sync_gradients_across_dp(dp_pg=dp_pg, reduce_op=reduce_op, **sync_options)
        return

    # Sync gradients (skip expert params -- those are reduced on expert_dp_pg).
    for name, param in module.named_parameters():
        if is_expert_param(param):
            continue
        dist.all_reduce(param.grad, op=reduce_op, group=dp_pg)


def sync_expert_gradients(
    module: nn.Module,
    expert_dp_pg: dist.ProcessGroup,
    reduce_op: dist.ReduceOp = dist.ReduceOp.AVG,
    grad_accumulator: Optional[GradientAccumulator] = None,
):
    """All-reduce expert-parameter gradients across the expert-data-parallel group.

    Expert parameters are those marked via `param._is_expert = True` by the trainer
    when wrapping the model. When `expert_dp_pg.size() == 1` there is no reduction
    to do and the call is a no-op.
    """
    if expert_dp_pg.size() == 1:
        return

    for name, param in module.named_parameters():
        if not is_expert_param(param):
            continue
        if grad_accumulator is not None:
            grad = grad_accumulator.get_grad_buffer(name=name)
        else:
            grad = param.grad
        if grad is None:
            continue
        dist.all_reduce(grad, op=reduce_op, group=expert_dp_pg)
