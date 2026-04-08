"""
ULFM pipeline engines with deferred gradient sync.

Early microbatches (0..N-2): model.no_sync() suppresses DDP hooks.
Last microbatch (N-1): ddp_trigger_sync_in_bwd() enables DDP hooks so
the deferred comm hook fires per-bucket, snapshots gradients, and returns
a pre-resolved Future (finalize_backward never blocks).

Actual ULFM allreduces are fired post-pipeline by the training manager.
"""

from typing import Optional

from torch.nn.parallel import DistributedDataParallel

from nanotron.optim.gradient_accumulator import GradientAccumulator
from nanotron.parallel.data_parallel.utils import ddp_trigger_sync_in_bwd
from nanotron.parallel.pipeline_parallel.engine import (
    AllForwardAllBackwardPipelineEngine,
    OneForwardOneBackwardPipelineEngine,
)
from nanotron.utils import ContextManagers
from torch import nn as torch_nn


class ULFMOneForwardOneBackwardPipelineEngine(OneForwardOneBackwardPipelineEngine):
    """1F1B pipeline engine with deferred ULFM gradient sync.

    Early microbatches use model.no_sync() so DDP hooks never fire.
    Last microbatch uses ddp_trigger_sync_in_bwd() so the deferred hook
    captures per-bucket snapshots and returns pre-resolved Futures.
    """

    def _get_bwd_context(
        self,
        model: torch_nn.Module,
        nb_backwards: int,
        grad_accumulator: Optional[GradientAccumulator],
    ):
        assert (
            self.nb_microbatches is not None
        ), "You must call `train_batch_iter` first and set `self.nb_microbatches`"
        is_ddp = isinstance(model, DistributedDataParallel)
        context_list = []
        if is_ddp:
            if nb_backwards < self.nb_microbatches - 1:
                context_list.append(model.no_sync())
                if grad_accumulator is not None:
                    context_list.append(grad_accumulator.no_sync())
            else:
                # Last microbatch: DDP hooks fire → deferred hook snapshots + queues
                context_list.append(ddp_trigger_sync_in_bwd(model_ddp=model))
        return ContextManagers(context_list)


class ULFMAllForwardAllBackwardPipelineEngine(AllForwardAllBackwardPipelineEngine):
    """AFAB pipeline engine with deferred ULFM gradient sync."""

    def _get_bwd_context(
        self,
        model: torch_nn.Module,
        nb_backwards: int,
        grad_accumulator: Optional[GradientAccumulator],
    ):
        assert (
            self.nb_microbatches is not None
        ), "You must call `train_batch_iter` first and set `self.nb_microbatches`"
        is_ddp = isinstance(model, DistributedDataParallel)
        context_list = []
        if is_ddp:
            if nb_backwards < self.nb_microbatches - 1:
                context_list.append(model.no_sync())
                if grad_accumulator is not None:
                    context_list.append(grad_accumulator.no_sync())
            else:
                # Last microbatch: DDP hooks fire → deferred hook snapshots + queues
                context_list.append(ddp_trigger_sync_in_bwd(model_ddp=model))
        return ContextManagers(context_list)
