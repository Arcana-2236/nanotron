---
library_name: nanotron
---

# LlaMoE

Modeling code for LlaMoE to use with [Nanotron](https://github.com/huggingface/nanotron/)

## 🚀 Quickstart

```bash
# Generate a config file
python examples/moe/config_llamoe.py

# Install megablocks
pip install megablocks

# Run training
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=4 examples/moe/train_moe.py --config-file examples/moe/config_llamoe.yaml
```

## 🚀 Use your custom model
- Update the `LlaMoEConfig` class in `config_llamoe.py` to match your model's configuration
- Update the `LlaMoEForTraining` class in `modeling_llamoe.py` to match your model's architecture
- Pass the previous to the `DistributedTrainer` class in `train_moe.py`:
```python
trainer = DistributedTrainer(config_file, model_class=LlaMoEForTraining, model_config_class=LlaMoEConfig)
```
- Run training as usual


## Parallelism semantics (post 2026-05-11 refactor)

`expert_parallel_size` now subdivides DP rather than adding a separate axis. The world size
formula is `tp × pp × dp` (NOT `× expert_parallel_size`). With `dp=4, expert_parallel_size=4`,
training uses 4 GPUs — each rank holds a unique data shard *and* one (or more, if
`moe_num_experts > expert_parallel_size`) expert(s). All-to-all token routing happens inside
the EP sub-group of `dp_pg`.

Constraints:
- `dp % expert_parallel_size == 0`.
- `moe_num_experts % expert_parallel_size == 0` (pre-existing megablocks constraint).

Old checkpoints from before this refactor need conversion:

```
python tools/convert_checkpoint_ep_subset.py \
    --src OLD_CHECKPOINT_DIR --dst NEW_CHECKPOINT_DIR \
    --dp_old <old DP> --ep <expert_parallel_size>
```

Per-device FLOPs/s reported by `get_flops_per_sec` will increase compared to the old layout:
the same model FLOPs are now amortized over fewer devices, which is the expected (correct)
behavior.

## Credits
Credits to the following repositories from which the code was adapted:
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py
- https://github.com/stanford-futuredata/megablocks/blob/main/megablocks/layers/dmoe.py
