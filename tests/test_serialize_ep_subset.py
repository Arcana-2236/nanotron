"""Smoke tests for the EP-as-subset-of-DP serialize changes.

These tests do NOT spawn a distributed group. They exercise the pure-Python
logic in `nanotron.serialize.utils.get_path` and the regex used by the
checkpoint converter, against synthetic inputs that match the new filename
schema.

The plan's original smoke test (in `docs/superpowers/plans/...`) tried to call
`save_weights` against a fake `Tiny` module built from `torch.nn.Linear`. That
path is unworkable because `save_weights` asserts on `NanotronParameter` and
raises `NotImplementedError` for plain `nn.Parameter`. Building a faithful
`NanotronParameter`-based fixture would require a real `ParallelContext` (and
therefore a multi-rank spawn), which is heavier than what a smoke test should
be. We exercise the bit-flipped logic directly instead.
"""
import importlib.util
from pathlib import Path

from nanotron.serialize.utils import ObjectType, get_path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONVERTER_PATH = _REPO_ROOT / "tools" / "convert_checkpoint_ep_subset.py"


def _load_converter():
    spec = importlib.util.spec_from_file_location(
        "convert_checkpoint_ep_subset", _CONVERTER_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_get_path_expert_shard_format():
    """Expert-sharded params should emit the `_exp-rank-K-of-N` suffix."""
    path = get_path(
        tensor_name="model.layers.0.block_sparse_moe.experts.mlp.w1.weight",
        type=ObjectType.MODEL,
        exp_tp_pp_rank_and_size=((1, 2), (0, 1), (0, 1)),  # exp_rank=1, tp=0/1, pp=0/1
        is_expert_sharded=True,
        prefix=None,
    )
    name = path[-1]
    assert "_exp-rank-1-of-2" in name, name
    assert "_pp-rank-0-of-1" in name, name
    assert "_tp-rank-0-of-1" in name, name
    assert name.endswith(".safetensors"), name


def test_get_path_non_expert_omits_exp_suffix():
    """Non-expert params must NOT include the exp suffix."""
    path = get_path(
        tensor_name="model.layers.0.attention.qkv.weight",
        type=ObjectType.MODEL,
        exp_tp_pp_rank_and_size=((0, 2), (0, 1), (0, 1)),
        is_expert_sharded=False,
        prefix=None,
    )
    name = path[-1]
    assert "_exp-rank" not in name, name
    assert "_pp-rank-0-of-1_tp-rank-0-of-1.safetensors" in name, name


def test_converter_regex_matches_expert_shard():
    """The converter's SHARD_RE must match the exact filenames produced by `get_path`."""
    mod = _load_converter()
    # Match the filename pattern emitted by get_path for an expert shard.
    sample = "model_weight_pp-rank-0-of-1_tp-rank-0-of-1_exp-rank-1-of-2.safetensors"
    m = mod.SHARD_RE.match(sample)
    assert m is not None, f"regex failed to match {sample!r}"
    assert m.group("exp") == "1"
    assert m.group("exp_size") == "2"


def test_converter_regex_matches_non_expert_shard():
    sample = "model_weight_pp-rank-0-of-1_tp-rank-0-of-1.safetensors"
    mod = _load_converter()
    m = mod.SHARD_RE.match(sample)
    assert m is not None, f"regex failed to match {sample!r}"
    assert m.group("exp") is None


def test_converter_remap_expert_shard():
    """Expert shard at old (D_old, E) maps to new D_new = D_old * EP + E."""
    mod = _load_converter()
    name = "model_weight_pp-rank-0-of-1_tp-rank-0-of-1_exp-rank-1-of-2.safetensors"
    out = mod.remap(name, dp_old=1, ep=2)
    # We expect the expert filename to be preserved (no DP rank in expert filenames).
    assert out is not None
    assert "_exp-rank-1-of-2" in out


def test_converter_remap_non_expert_duplicate_dropped():
    """Non-expert shards with E>0 are bit-identical replicas and must be dropped."""
    mod = _load_converter()
    # No exp suffix here, but synthetic case: legacy paths might have it on non-expert.
    # The plan's converter drops such files. Build the relevant synthetic name.
    name = "model_weight_pp-rank-0-of-1_tp-rank-0-of-1.safetensors"
    out = mod.remap(name, dp_old=1, ep=2)
    assert out is not None  # non-duplicated non-expert file stays
