"""One-shot converter: orthogonal-EP layout (TP*PP*DP*EP world) to
EP-as-subset-of-DP layout (TP*PP*DP_new world with DP_new = DP_old * EP).

Usage:
    python tools/convert_checkpoint_ep_subset.py \
        --src /path/to/old/checkpoint \
        --dst /path/to/new/checkpoint \
        --dp_old 4 --ep 4

Per spec section 3.7, expert shards at old (D_old, E) move to new
D_new = D_old * EP + E. Non-expert shards at old (D_old=*, E=0) move to new
D_new = D_old * EP (the other E>0 non-expert shards are bit-identical replicas
the old saver already skipped, so they should not exist on disk; we ignore
them defensively if they do).
"""
import argparse
import re
import shutil
from pathlib import Path


SHARD_RE = re.compile(
    r"(?P<base>.*)_pp-rank-(?P<pp>\d+)-of-(?P<pp_size>\d+)"
    r"(?:_dp-(?P<dp>\d+)-of-(?P<dp_size>\d+))?"
    r"_tp-rank-(?P<tp>\d+)-of-(?P<tp_size>\d+)"
    r"(?:_exp-rank-(?P<exp>\d+)-of-(?P<exp_size>\d+))?"
    r"\.(?P<ext>safetensors|pt)$"
)


def remap(name: str, dp_old: int, ep: int):
    """Return the new filename, or None if this shard should be dropped (duplicate)."""
    m = SHARD_RE.match(name)
    if not m:
        return name  # leave non-shard files unchanged
    parts = m.groupdict()
    pp = int(parts["pp"])
    pp_size = int(parts["pp_size"])
    tp = int(parts["tp"])
    tp_size = int(parts["tp_size"])
    d_old = int(parts["dp"]) if parts["dp"] is not None else 0
    e_old = int(parts["exp"]) if parts["exp"] is not None else 0
    is_expert = parts["exp"] is not None

    if is_expert:
        d_new = d_old * ep + e_old  # noqa: F841  (kept for documentation / future use)
    else:
        if e_old != 0:
            return None  # bit-identical replica; drop
        d_new = d_old * ep  # noqa: F841

    base = parts["base"]
    ext = parts["ext"]
    if is_expert:
        # Expert filenames preserve the exp-rank-of-EP suffix; EP is unchanged.
        return f"{base}_pp-rank-{pp}-of-{pp_size}_tp-rank-{tp}-of-{tp_size}_exp-rank-{e_old}-of-{ep}.{ext}"
    else:
        # Non-expert filenames drop any exp suffix (get_path omits it for non-expert).
        return f"{base}_pp-rank-{pp}-of-{pp_size}_tp-rank-{tp}-of-{tp_size}.{ext}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--dp_old", type=int, required=True, help="Old DP size (orthogonal layout).")
    ap.add_argument("--ep", type=int, required=True, help="EP size (unchanged).")
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    n_copied = n_dropped = 0
    for path in args.src.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(args.src)
        new_name = remap(rel.name, args.dp_old, args.ep)
        if new_name is None:
            n_dropped += 1
            continue
        out = args.dst / rel.parent / new_name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
        n_copied += 1
    print(f"copied {n_copied} shards, dropped {n_dropped} duplicates")


if __name__ == "__main__":
    main()
