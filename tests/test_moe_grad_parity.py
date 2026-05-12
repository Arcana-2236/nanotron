"""Parity test: DP=4, EP=2 MoE step matches DP=1, EP=1 with equivalent grad-accum.

Skipped unless 4 GPUs + megablocks/stk are available.
"""
import importlib.util
import json
import os
import tempfile

import pytest
import torch


def _have_megablocks() -> bool:
    return importlib.util.find_spec("megablocks") is not None and importlib.util.find_spec("stk") is not None


from helpers.utils import available_gpus, init_distributed, rerun_if_address_is_in_use


def _one_step_loss(parallel_context, seed: int, micro_batch: int, full_batch_seed: int):
    """Run one forward/backward on a tiny LlaMoE; return the loss tensor's float value
    after the post-backward grad reduce."""
    import sys

    sys.path.insert(0, "examples/moe")
    from config_llamoe import LlaMoEConfig
    from llamoe import LlaMoEForTraining
    from nanotron import distributed as dist
    from nanotron.config import ParallelismArgs
    from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode

    torch.manual_seed(seed)
    config = LlaMoEConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        moe_num_experts=4,
        num_experts_per_tok=1,
        rms_norm_eps=1e-5,
        pad_token_id=0,
    )
    parallel_config = ParallelismArgs(
        dp=parallel_context.data_parallel_size,
        pp=parallel_context.pipeline_parallel_size,
        tp=parallel_context.tensor_parallel_size,
        expert_parallel_size=parallel_context.expert_parallel_size,
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
    )
    model = LlaMoEForTraining(
        config=config,
        parallel_context=parallel_context,
        parallel_config=parallel_config,
    ).cuda()
    # init_model_randomly expects a nested config object exposing
    # `config.model.init_method.std` and `config.model.model_config`.
    # We build the minimal stub it needs via anonymous type chains rather
    # than importing the full Config dataclass machinery.
    model.init_model_randomly(
        config=type(
            "X",
            (),
            {
                "model": type(
                    "Y",
                    (),
                    {
                        "init_method": type("Z", (), {"std": 0.02})(),
                        "model_config": config,
                    },
                )()
            },
        )()
    )

    # Per-rank data shard from a single deterministic full-batch source.
    full_batch = (
        torch.arange(parallel_context.data_parallel_size * micro_batch * 16).reshape(
            parallel_context.data_parallel_size * micro_batch, 16
        )
        % config.vocab_size
    )
    dp_rank = dist.get_rank(parallel_context.dp_pg)
    local = full_batch[dp_rank * micro_batch : (dp_rank + 1) * micro_batch].cuda()

    out = model(
        input_ids=local,
        input_mask=torch.ones_like(local, dtype=torch.bool),
        label_ids=local,
        label_mask=torch.ones_like(local, dtype=torch.bool),
    )
    out["loss"].backward()
    # Aggregate the per-rank loss over DP to get the global loss.
    loss = out["loss"].detach()
    dist.all_reduce(loss, group=parallel_context.dp_pg, op=dist.ReduceOp.AVG)
    return loss.item()


def _collect_loss_to_file(parallel_context, out_path: str):
    """Compute the parity loss and have global rank 0 write it to ``out_path``.

    We persist via the filesystem because ``mp.spawn`` runs each rank in a fresh
    subprocess, so any Python-level dict the parent test holds is not visible to
    the closure executing inside the spawned workers.
    """
    from nanotron import distributed as dist

    loss_value = _one_step_loss(
        parallel_context, seed=1234, micro_batch=2, full_batch_seed=1234
    )
    if dist.get_rank(parallel_context.world_pg) == 0:
        with open(out_path, "w") as f:
            json.dump({"loss": loss_value}, f)


@pytest.mark.skipif(
    not _have_megablocks() or available_gpus() < 4,
    reason="needs 4 gpus + megablocks/stk",
)
@rerun_if_address_is_in_use()
def test_loss_parity_dp4_ep2_vs_dp1_ep1():
    with tempfile.TemporaryDirectory() as tmpdir:
        dp4_ep2_path = os.path.join(tmpdir, "dp4_ep2.json")
        dp1_ep1_path = os.path.join(tmpdir, "dp1_ep1.json")

        init_distributed(tp=1, dp=4, pp=1, ep=2)(_collect_loss_to_file)(
            out_path=dp4_ep2_path
        )
        init_distributed(tp=1, dp=1, pp=1, ep=1)(_collect_loss_to_file)(
            out_path=dp1_ep1_path
        )

        with open(dp4_ep2_path) as f:
            loss_dp4_ep2 = json.load(f)["loss"]
        with open(dp1_ep1_path) as f:
            loss_dp1_ep1 = json.load(f)["loss"]

    # Allow 1e-4 relative tolerance — fp32 accumulation differences between an
    # 8-token global batch processed 1x vs 4x (2 each) typically agree to ~1e-5.
    assert (
        abs(loss_dp4_ep2 - loss_dp1_ep1) <= 1e-4 * abs(loss_dp1_ep1) + 1e-6
    ), f"loss parity failed: dp4_ep2={loss_dp4_ep2} dp1_ep1={loss_dp1_ep1}"
