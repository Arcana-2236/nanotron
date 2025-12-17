"""
0. Work directory
cd /workspace/cola_nanotron/nanotron

1. Generate a config file
python examples/cola/config_cola_llama.py

2. Run the training
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 examples/cola/train_vanilla_cola.py --config-file examples/cola/config_cola_llama_7b.yaml
"""
import argparse
import os
import sys


from nanotron import logging
from nanotron.trainer import DistributedTrainer
from config_cola_llama import ColaLlamaConfig
from vanilla_cola_llama import VanillaColaLlamaForTraining

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from run_train import get_dataloader  # noqa

logger = logging.get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # Load trainer and data
    trainer = DistributedTrainer(config_file, model_config_class=ColaLlamaConfig, model_class=VanillaColaLlamaForTraining)
    # trainer = DistributedTrainer(config_file)
    dataloader = get_dataloader(trainer)

    # Train
    trainer.train(dataloader)