"""
LlaMoE training script.

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc_per_node=4 examples/moe/train_moe.py --config-file examples/moe/config_llamoe.yaml
```
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from config_llamoe import LlaMoEConfig
from llamoe import LlaMoEForTraining
from nanotron import logging
from nanotron.config import get_config_from_file, apply_config_overrides
from nanotron.trainer import DistributedTrainer
from nanotron.utils import get_args

from run_train import get_dataloader  # noqa

logger = logging.get_logger(__name__)


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # Load config with MoE-specific model config class
    config = get_config_from_file(config_file, model_config_class=LlaMoEConfig)

    # Apply command line overrides
    config = apply_config_overrides(config, args)

    # Apply generic dotted-path overrides (e.g. --override optimizer.optimizer_factory.muon_mode=sgd)
    if args.override:
        from nanotron.config.config import apply_generic_overrides
        config = apply_generic_overrides(config, args.override)

    # Load trainer and data (pass model_class so DistributedTrainer maps LlaMoEConfig -> LlaMoEForTraining)
    trainer = DistributedTrainer(config, model_class=LlaMoEForTraining)
    dataloader = get_dataloader(trainer)

    # Train
    trainer.train(dataloader)
