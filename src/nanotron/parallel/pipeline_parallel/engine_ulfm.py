"""
ULFM pipeline engines with deferred gradient sync.

Early microbatches (0..N-2): model.no_sync() suppresses DDP hooks.
Last microbatch (N-1): ddp_trigger_sync_in_bwd() enables DDP hooks so
the deferred comm hook fires per-bucket, snapshots gradients, and returns
a pre-resolved Future (finalize_backward never blocks).

Actual ULFM allreduces are fired post-pipeline by the training manager.

Callbacks (set by trainer for extended passes at policy boundaries):
  should_zero_loss_fn: Called per-forward; if True, loss *= 0 (rank shouldn't contribute)
  pre_last_backward_fn: Called before last microbatch's backward (wait for async restore)
"""

from typing import Callable, Dict, Iterable, Optional, Union

import torch
from torch import nn as torch_nn
from torch.nn.parallel import DistributedDataParallel

from nanotron import logging
from nanotron.logging import log_rank
from nanotron.optim.gradient_accumulator import GradientAccumulator
from nanotron.parallel.data_parallel.utils import ddp_trigger_sync_in_bwd
from nanotron.parallel.pipeline_parallel.context_manager import attach_pipeline_state_to_model
from nanotron.parallel.pipeline_parallel.engine import (
    AllForwardAllBackwardPipelineEngine,
    OneForwardOneBackwardPipelineEngine,
    PipelineEngine,
)
from nanotron.parallel.pipeline_parallel.state import PipelineEvalBatchState, PipelineTrainBatchState
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.utils import ContextManagers

logger = logging.get_logger(__name__)


class _ULFMEngineMixin:
    """Mixin adding ULFM callbacks to pipeline engines.

    Attributes set by the trainer before each train_batch_iter call:
      should_zero_loss_fn: If set and returns True, multiply loss by 0
          (rank should not contribute gradients for this microbatch).
      pre_last_backward_fn: If set, called before the last microbatch's
          backward to wait for async gradient restoration.
    """

    should_zero_loss_fn: Optional[Callable[[], bool]] = None
    pre_last_backward_fn: Optional[Callable[[], None]] = None

    def forward(
        self,
        context,
        state: PipelineTrainBatchState,
        micro_batch: Dict[str, Union[torch.Tensor, TensorPointer]],
        model: torch_nn.Module,
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # Replicate base forward() but skip per-microbatch loss division.
        # ULFM normalizes once before optimizer step via normalize_and_step().
        if hasattr(state, "nb_forwards"):
            state.nb_forwards += 1
            log_rank(
                f"Forward micro batch id: {state.nb_forwards}",
                logger=logger,
                level=logging.DEBUG,
            )
        
        state.new_micro_batch_forward()
        with context:
            output = model(**micro_batch)
        
        if not isinstance(output, dict):
            output = {"loss": output}
        
        if not isinstance(output["loss"], TensorPointer):
            if self.should_zero_loss_fn is not None and self.should_zero_loss_fn():
                output["loss"] = output["loss"] * 0.0
        
        if not isinstance(output["loss"], TensorPointer) and output["loss"].requires_grad:
            state.register_activation_requiring_backward(output["loss"])
        return output

    def validate_batch_iter(
        self,
        model: torch_nn.Module,
        batch: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
        nb_microbatches: int,
    ) -> Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]]:
        """Validation uses PipelineEngine.forward() — includes per-microbatch
        loss division and skips ULFM callbacks (no contribution tracking)."""
        state = PipelineEvalBatchState()
        self.nb_microbatches = nb_microbatches

        outputs = []
        with attach_pipeline_state_to_model(model=model, pipeline_state=state):
            for micro_batch in batch:
                context = self._get_fwd_context(model=model)
                # Use base forward (with /nb_microbatches, no ULFM callbacks)
                output = PipelineEngine.forward(self, context=context, state=state, micro_batch=micro_batch, model=model)
                for _ in range(len(state.microbatches_activations_to_send)):
                    send_activation = state.microbatches_activations_to_send.popleft()
                    send_activation()
                if not isinstance(output, dict):
                    output = {"loss": output}
                if not isinstance(output["loss"], TensorPointer):
                    output = {k: v.detach() for k, v in output.items()}
                outputs.append(output)

        return outputs


class ULFMOneForwardOneBackwardPipelineEngine(
    _ULFMEngineMixin, OneForwardOneBackwardPipelineEngine
):
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
                # Last microbatch: wait for async restore, then enable DDP hooks
                if self.pre_last_backward_fn is not None:
                    self.pre_last_backward_fn()
                context_list.append(ddp_trigger_sync_in_bwd(model_ddp=model))
        return ContextManagers(context_list)


class ULFMAllForwardAllBackwardPipelineEngine(
    _ULFMEngineMixin, AllForwardAllBackwardPipelineEngine
):
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
                # Last microbatch: wait for async restore, then enable DDP hooks
                if self.pre_last_backward_fn is not None:
                    self.pre_last_backward_fn()
                context_list.append(ddp_trigger_sync_in_bwd(model_ddp=model))
        return ContextManagers(context_list)
