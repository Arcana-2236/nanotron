"""
ULFM pipeline engines: always-no_sync for deferred gradient sync.

DDP hooks never fire during the pipeline. Gradients accumulate locally
across all microbatches. After the pipeline completes, the training
manager allreduces gradients per-bucket via ULFM.
"""

from typing import Optional

from torch.nn.parallel import DistributedDataParallel

from nanotron.optim.gradient_accumulator import GradientAccumulator
from nanotron.parallel.pipeline_parallel.engine import (
    AllForwardAllBackwardPipelineEngine,
    OneForwardOneBackwardPipelineEngine,
)
from nanotron.utils import ContextManagers
from torch import nn as torch_nn


class ULFMOneForwardOneBackwardPipelineEngine(OneForwardOneBackwardPipelineEngine):
    """1F1B pipeline engine that always suppresses DDP gradient sync.

    All microbatches use model.no_sync() so DDP hooks never fire and
    finalize_backward never blocks on MPI futures.
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
            context_list.append(model.no_sync())
            if grad_accumulator is not None and nb_backwards < self.nb_microbatches - 1:
                context_list.append(grad_accumulator.no_sync())
        return ContextManagers(context_list)


class ULFMAllForwardAllBackwardPipelineEngine(AllForwardAllBackwardPipelineEngine):
    """AFAB pipeline engine that always suppresses DDP gradient sync."""

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
            context_list.append(model.no_sync())
            if grad_accumulator is not None and nb_backwards < self.nb_microbatches - 1:
                context_list.append(grad_accumulator.no_sync())
        return ContextManagers(context_list)
