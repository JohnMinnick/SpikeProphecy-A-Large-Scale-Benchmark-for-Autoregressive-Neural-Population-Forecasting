"""Corrected local eval for a teacher model (Mamba/LRU/Transformer/LSTM).

Mirror of scripts/eval_local_corrected.py but loads via create_teacher_model.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.common import create_teacher_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.device import resolve_device  # noqa: E402
from src.data.multi_session_loader import (  # noqa: E402
    MaskedSpikeCountDataset,
    pad_to_channels,
    build_channel_mask,
)
from scripts.eval_local_corrected import session_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--teacher-config", type=str, required=True)
    p.add_argument(
        "--cache-dir",
        type=str,
        default=str(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "combined_steinmetz_ibl_cache"
        ),
    )
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--max-sessions", type=int, default=None)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device()
    print(f"Device: {device}")

    cache_dir = Path(args.cache_dir)
    metadata = json.load(open(cache_dir / "metadata.json"))
    m_max = metadata["m_max"]
    history_bins = metadata.get("history_bins", 10)
    n_sessions = metadata["num_sessions"]
    print(
        f"Cache: {cache_dir} | M_max={m_max} | "
        f"sessions={n_sessions} | history={history_bins}"
    )

    # Build teacher from config
    teacher_cfg = load_config(args.teacher_config)
    # The teacher config wraps under 'teacher' key
    cfg_for_factory = teacher_cfg.get("teacher", teacher_cfg)
    model = create_teacher_model(
        config=cfg_for_factory,
        input_size=m_max,
        session_dims=None,  # global head, no session-specific dims
    )

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        print(f"strict=True failed: {e}\nFalling back to strict=False")
        model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Loaded {args.ckpt}: epoch={ckpt.get('epoch', '?')}, "
        f"params={n_params:,}"
    )

    sessions_to_run = (
        n_sessions if args.max_sessions is None
        else min(args.max_sessions, n_sessions)
    )
    print(f"Evaluating {sessions_to_run} sessions, split={args.split}")

    per_session_results = []
    t0 = time.time()
    for sess_idx in range(sessions_to_run):
        sess_info = metadata["sessions"][sess_idx]
        m_i = sess_info["num_units"]
        npy_path = cache_dir / f"session_{sess_idx:03d}.npy"
        if not npy_path.exists():
            continue

        counts = np.load(npy_path)
        bounds = sess_info["split_boundaries"]
        if args.split == "val":
            split_start = bounds["train_end"]
            split_end = bounds["val_end"]
        else:
            split_start = bounds["val_end"]
            split_end = sess_info["num_bins"]

        split_counts = counts[:, split_start:split_end].astype(np.int32)
        del counts
        if split_counts.shape[1] <= history_bins:
            continue

        padded = pad_to_channels(split_counts, m_max)
        del split_counts
        mask_index = np.zeros(padded.shape[1], dtype=np.int32)
        ds = MaskedSpikeCountDataset(
            spike_counts=padded,
            mask_index=mask_index,
            session_masks=build_channel_mask(m_i, m_max).reshape(1, -1),
            history_bins=history_bins,
            output_channels=m_max,
        )
        del padded
        if len(ds) == 0:
            continue

        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        all_pred = []
        all_targ = []
        for batch in dl:
            x = batch[0].to(device)
            y = batch[1]
            out = model(x)
            if isinstance(out, dict):
                rates = out.get("rates", out.get("output"))
            elif isinstance(out, tuple):
                rates = out[0]
            else:
                rates = out
            all_pred.append(rates[:, :m_i].cpu())
            all_targ.append(y[:, :m_i])

        preds = torch.cat(all_pred, dim=0)
        targs = torch.cat(all_targ, dim=0)

        m = session_metrics(preds, targs)
        m["session_idx"] = sess_idx
        m["source"] = sess_info.get("source", "unknown")
        m["n_neurons"] = m_i
        m["n_test_bins"] = int(preds.shape[0])
        m["brain_regions"] = sess_info.get("brain_regions")
        per_session_results.append(m)

        elapsed = time.time() - t0
        print(
            f"  s{sess_idx:03d} ({m['source']:>9}, m_i={m_i:4d}, "
            f"T={preds.shape[0]:5d}): "
            f"per_ch={m['per_ch_r']:.4f} weighted={m['weighted_r']:.4f} "
            f"global={m['global_r']:.4f} pop={m['pop_rate_r']:.4f} "
            f"spat={m['spatial_r']:.4f} cos={m['cosine_sim']:.4f} "
            f"mae={m['mae']:.4f} | {elapsed:.0f}s"
        )

    if not per_session_results:
        print("No sessions evaluated")
        return

    total_neurons = sum(s["n_neurons"] for s in per_session_results)
    total_bins = sum(s["n_test_bins"] for s in per_session_results)

    def neuron_weighted(key: str) -> float:
        return float(
            sum(s[key] * s["n_neurons"] for s in per_session_results)
            / max(total_neurons, 1)
        )

    aggregate = {
        "n_sessions": len(per_session_results),
        "total_neurons": total_neurons,
        "total_test_bins": total_bins,
        "neuron_weighted": {
            k: neuron_weighted(k)
            for k in [
                "per_ch_r",
                "weighted_r",
                "global_r",
                "pop_rate_r",
                "spatial_r",
                "cosine_sim",
                "mae",
            ]
        },
    }

    by_source = {}
    for src in {s["source"] for s in per_session_results}:
        ss = [s for s in per_session_results if s["source"] == src]
        n = sum(s["n_neurons"] for s in ss)
        by_source[src] = {
            "n_sessions": len(ss),
            "total_neurons": n,
            "neuron_weighted": {
                k: float(sum(s[k] * s["n_neurons"] for s in ss) / max(n, 1))
                for k in [
                    "per_ch_r",
                    "weighted_r",
                    "global_r",
                    "pop_rate_r",
                    "spatial_r",
                    "cosine_sim",
                    "mae",
                ]
            },
        }
    aggregate["by_source"] = by_source

    out = {
        "checkpoint": args.ckpt,
        "epoch": ckpt.get("epoch", None),
        "n_params": int(n_params),
        "split": args.split,
        "aggregate": aggregate,
        "per_session": per_session_results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    elapsed = time.time() - t0
    print()
    print("=" * 72)
    print(f"  AGGREGATE TEACHER — {args.ckpt}")
    print("=" * 72)
    print(
        f"  Sessions: {aggregate['n_sessions']}, "
        f"neurons: {aggregate['total_neurons']:,}, "
        f"bins: {aggregate['total_test_bins']:,}"
    )
    nw = aggregate["neuron_weighted"]
    print(
        f"  per_ch_r   = {nw['per_ch_r']:.4f}    "
        f"weighted_r = {nw['weighted_r']:.4f}    "
        f"global_r   = {nw['global_r']:.4f}"
    )
    print(
        f"  pop_rate_r = {nw['pop_rate_r']:.4f}    "
        f"spatial_r  = {nw['spatial_r']:.4f}    "
        f"cosine_sim = {nw['cosine_sim']:.4f}    "
        f"mae = {nw['mae']:.4f}"
    )
    for src, sd_ in by_source.items():
        snw = sd_["neuron_weighted"]
        print(
            f"  {src:>9}: n={sd_['n_sessions']:3d} "
            f"neur={sd_['total_neurons']:6,} | "
            f"per_ch={snw['per_ch_r']:.4f} "
            f"global={snw['global_r']:.4f} "
            f"pop={snw['pop_rate_r']:.4f} "
            f"cos={snw['cosine_sim']:.4f}"
        )
    print(f"  Wrote: {out_path}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
