# src/nanotron/trainer_ulfm.py
"""
ULFMDistributedTrainer: Nanotron DistributedTrainer with ULFM fault-tolerant DP.

Changes from parent:
  - _create_parallel_context(): returns ULFMParallelContext (ULFM dp_pg, NCCL tp/pp/ep)
  - init_model(): after parent builds DDP model, creates NanotronULFMTrainingManager
    which registers the ULFM comm hook on the existing DDP model
  - training_step(): calls prepare_iteration()/after_pipeline() around pipeline engine;
    skips optimizer/lr-scheduler on failure iterations
  - train(): calls failure_simulator.begin_minibatch() each step; skips
    logging/checkpoint when training_step signals a failure iteration

All ULFM recovery logic lives in NanotronULFMTrainingManager / StepTxnOrchestrator /
policy; this class only manages the nanotron-side lifecycle.

Note: The `train()` override does not include the parent's profiler (`get_profiler`),
NVTX ranges, or CUDA profiler start/stop. These are correctness-neutral omissions;
nanotron profiling tools will not capture ULFM runs. TODO: add profiler support.
"""

import gc
import os
import sys
import time
from typing import Dict, Iterable, Iterator, Optional, Tuple, Union

import torch
from torch.nn.parallel import DistributedDataParallel

# ULFM modules live one directory above the nanotron package root
# trainer_ulfm.py is in nanotron/src/nanotron/ → 3 up → mpi_ulfm_extension/
_ULFM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ULFM_ROOT not in sys.path:
    sys.path.insert(0, _ULFM_ROOT)

from nanotron import distributed as dist
from nanotron import logging as nanotron_logging
from nanotron.models import NanotronModel
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.context_ulfm import ULFMParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.tied_parameters import sync_tied_weights_gradients
from nanotron.sanity_checks import (
    after_optim_step_sanity_checks,
    after_tbi_sanity_checks,
    before_optim_step_sanity_checks,
    before_tbi_sanity_checks,
)
from nanotron.trainer import DistributedTrainer
from nanotron.training_manager_nanotron import NanotronULFMTrainingManager
from nanotron.logging import log_memory
from nanotron.optim.clip_grads import clip_grad_norm
from nanotron.serialize.optimizer import state_dict_to_device

from ulfm_collectives.failure_simulator import FailureSimulator, set_failure_simulator

logger = nanotron_logging.get_logger(__name__)


class ULFMDistributedTrainer(DistributedTrainer):
    """
    Nanotron trainer with ULFM fault-tolerant data parallelism.

    Constructor args (in addition to DistributedTrainer args):
      failure_simulator: Optional[FailureSimulator]
          Pre-configured FailureSimulator. If None, no failure injection.
      ulfm_policy_type: str (default "static")
          Policy type forwarded to NanotronULFMTrainingManager.
    """

    def __init__(
        self,
        config_or_config_file,
        failure_simulator: Optional[FailureSimulator] = None,
        ulfm_policy_type: str = "static",
        **kwargs,
    ):
        self._failure_simulator = failure_simulator
        self._ulfm_policy_type = ulfm_policy_type
        # ulfm_manager is set in init_model(); declare early for type checkers
        self.ulfm_manager: Optional[NanotronULFMTrainingManager] = None

        super().__init__(config_or_config_file, **kwargs)

        # Register failure simulator globally (ulfm_hook.py reads it via get_failure_simulator())
        if self._failure_simulator is not None:
            set_failure_simulator(self._failure_simulator)
            self._failure_simulator.initialize(
                rank=dist.get_rank(self.parallel_context.world_pg),
                world_size=dist.get_world_size(self.parallel_context.world_pg),
            )

    # ------------------------------------------------------------------
    # Override 1: use ULFMParallelContext
    # ------------------------------------------------------------------

    def _create_parallel_context(self) -> ParallelContext:
        return ULFMParallelContext(
            tensor_parallel_size=self.config.parallelism.tp,
            pipeline_parallel_size=self.config.parallelism.pp,
            data_parallel_size=self.config.parallelism.dp,
            expert_parallel_size=self.config.parallelism.expert_parallel_size,
        )

    # ------------------------------------------------------------------
    # Override 2: register ULFM hook on the DDP model nanotron creates
    # ------------------------------------------------------------------

    def init_model(self) -> Union[NanotronModel, DistributedDataParallel]:
        # Let parent build the model and wrap in DDP using self.parallel_context.dp_pg.
        # At this point dp_pg is a ProcessGroupULFM (from ULFMParallelContext).
        model = super().init_model()

        # model is DistributedDataParallel when dp > 1
        if isinstance(model, DistributedDataParallel):
            dp_size = self.parallel_context.data_parallel_size
            grad_accum = self.config.tokens.batch_accumulation_per_replica

            self.ulfm_manager = NanotronULFMTrainingManager(
                ddp_model=model,
                dp_pg=self.parallel_context.dp_pg,
                grad_accum_steps=grad_accum,
                policy_type=self._ulfm_policy_type,
                # StaticWorldPolicy requires initial_world_size
                initial_world_size=dp_size,
            )
        else:
            # dp_size == 1: single GPU, no ULFM needed
            logger.warning("dp_size=1, ULFM manager not created (single-GPU mode).")

        return model

    # ------------------------------------------------------------------
    # Override 3: training_step with ULFM lifecycle
    # ------------------------------------------------------------------

    def training_step(
        self, dataloader: Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]]
    ) -> Tuple[Iterable[Dict], Optional[torch.Tensor]]:
        """
        ULFM-aware training step.

        Structure:
          1. before_tbi sanity checks (same as parent)
          2. prepare_iteration(): ULFM pre-backward setup
          3. pipeline_engine.train_batch_iter(): forward + backward (ULFM hook fires here)
          4. after_tbi sanity checks (same as parent)
          5. sync_tied_weights_gradients(): NCCL, unaffected by DP failure
          6. clip_grad (if configured)
          7. before_optim_step sanity checks
          8. after_pipeline(): ULFM restore + optimizer.step() (or skip on failure)
          9. If stepped: after_optim_step sanity checks + lr_scheduler.step() + loss avg
        """
        before_tbi_sanity_checks(
            self.config,
            self.parallel_context,
            self.unwrapped_model,
            self.grad_accumulator,
            self.lr_scheduler,
        )

        if self.iteration_step < self.initial_iter_step + 5:
            log_memory(logger=logger)

        # Step 2: ULFM pre-backward setup
        if self.ulfm_manager is not None:
            self.ulfm_manager.prepare_iteration(
                iteration_step=self.iteration_step,
                n_microbatches=self.n_micro_batches_per_batch,
            )

        # Step 3: forward + backward (ULFM comm hook fires on last backward)
        # grad_accumulator=None: FP32 training; ULFM hook handles grad sync via DDP
        outputs = self.pipeline_engine.train_batch_iter(
            model=self.model,
            pg=self.parallel_context.pp_pg,
            batch=(next(dataloader) for _ in range(self.n_micro_batches_per_batch)),
            nb_microbatches=self.n_micro_batches_per_batch,
            grad_accumulator=None,
        )

        if self.iteration_step < self.initial_iter_step + 5:
            log_memory(logger=logger)

        after_tbi_sanity_checks(
            self.config, self.parallel_context, self.unwrapped_model, self.grad_accumulator
        )

        # Step 5: tied weights gradient sync (NCCL groups; not affected by DP failure)
        sync_tied_weights_gradients(
            module=self.unwrapped_model,
            parallel_context=self.parallel_context,
            grad_accumulator=self.grad_accumulator,
        )

        # Step 6: gradient clipping
        if self.config.optimizer.clip_grad is not None:
            named_parameters = [
                (name, param)
                for name, param in self.unwrapped_model.get_named_params_with_correct_tied()
                if param.requires_grad
            ]
            self.grad_norm_unclipped = clip_grad_norm(
                mp_pg=self.parallel_context.mp_pg,
                named_parameters=named_parameters,
                grad_accumulator=self.grad_accumulator,
                max_norm=self.config.optimizer.clip_grad,
            )

        before_optim_step_sanity_checks(
            self.config,
            self.parallel_context,
            self.unwrapped_model,
            self.grad_accumulator,
            self.optimizer,
        )

        # Move optimizer states to GPU on the first step of a checkpoint-resume run.
        # Must happen before optimizer.step() which is called inside after_pipeline().
        if (
            self.init_checkpoint_path is not None
            and self.config.checkpoints.load_optimizer
            and self.iteration_step == self.initial_iter_step
        ):
            state_dict_to_device(self.optimizer.state_dict(), "cuda")

        # Step 8: ULFM restore + optimizer (returns False on failure iteration)
        if self.ulfm_manager is not None:
            stepped = self.ulfm_manager.after_pipeline(self.optimizer)
        else:
            # Single-GPU fallback
            self.optimizer.step()
            self.optimizer.zero_grad()
            stepped = True

        if not stepped:
            # Failure detected; gradients were invalid. Communicator already repaired by hook.
            # Signal to train() to skip logging/checkpoint for this iteration.
            return [], None

        after_optim_step_sanity_checks(
            self.config, self.parallel_context, self.unwrapped_model, self.grad_accumulator
        )

        # LR scheduler advances only on successful optimizer steps
        self.lr_scheduler.step()

        # Loss average across DP (dp_pg is repaired after ULFM hook; safe to use).
        # TODO: overlap this all_reduce with the optimizer step (currently runs sequentially;
        # the parent hides latency by starting the async all_reduce before optimizer.step()).
        if outputs and isinstance(outputs[0]["loss"], torch.Tensor):
            loss_avg = torch.stack([out["loss"] for out in outputs]).sum()
            handle = dist.all_reduce(
                loss_avg,
                group=self.parallel_context.dp_pg,
                async_op=True,
                op=dist.ReduceOp.AVG,
            )
            handle.wait()
        else:
            loss_avg = None

        self.post_train_step()

        return outputs, loss_avg

    # ------------------------------------------------------------------
    # Override 4: train() — tick failure simulator; skip failed iterations
    # ------------------------------------------------------------------

    def train(
        self,
        dataloader_or_dls,
        **kwargs,
    ) -> None:
        """
        Wraps parent train() to:
          - Tick the failure simulator at the start of each iteration
          - Skip metadata update / logging / checkpointing on failure iterations
            (when training_step returns ([], None))
        """
        self.pre_training(**kwargs)

        if self.config.checkpoints.save_initial_state and self.init_checkpoint_path is None:
            self.save_checkpoint()

        self.pipeline_engine = self.config.parallelism.pp_engine
        self.pipeline_engine.nb_microbatches = self.n_micro_batches_per_batch

        self.unwrapped_model = (
            self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        )
        self.unwrapped_model.module_id_to_prefix = {
            id(module): f"{module_name}."
            for module_name, module in self.unwrapped_model.named_modules()
        }
        self.unwrapped_model.module_id_to_prefix[id(self.unwrapped_model)] = ""

        self.initial_iter_step = self.metadata.last_train_step + 1
        self.last_iter_step = self.config.tokens.train_steps

        gc.collect()
        torch.cuda.empty_cache()

        for self.iteration_step in range(self.initial_iter_step, self.last_iter_step + 1):
            self.iteration_start_time = time.time()
            self._update_dataloader_based_on_training_stages(dataloader_or_dls)

            # Tick failure simulator (mirrors sim.begin_minibatch in mpi_ulfm_extension/main.py)
            if self._failure_simulator is not None:
                self._failure_simulator.begin_minibatch(self.iteration_step)

            outputs, loss_avg = self.training_step(dataloader=self.current_dataloader)

            # Failure iteration: outputs=[], loss_avg=None → skip bookkeeping
            if not outputs and loss_avg is None:
                logger.warning(
                    f"[Step {self.iteration_step}] Failure detected — skipping metadata/log/checkpoint."
                )
                continue

            # Normal iteration bookkeeping (same as parent)
            self.metadata.consumed_train_samples += self.global_batch_size
            self.metadata.last_train_step = self.iteration_step
            self.metadata.data_stages[
                self.metadata.last_stage_idx
            ].consumed_train_samples += self.global_batch_size

            if (self.iteration_step - 1) % self.config.logging.iteration_step_info_interval == 0:
                self.train_step_logs(outputs=outputs, loss_avg=loss_avg)

            if self.iteration_step % self.config.checkpoints.checkpoint_interval == 0:
                self.save_checkpoint()

            if (
                self.config.tokens.val_check_interval > 0
                and self._val_dataloader_factory is not None
                and self.iteration_step % self.config.tokens.val_check_interval == 0
            ):
                self.run_validation()

        dist.barrier()

        if self.config.checkpoints.save_final_state:
            self.save_checkpoint()

        self.post_training()
