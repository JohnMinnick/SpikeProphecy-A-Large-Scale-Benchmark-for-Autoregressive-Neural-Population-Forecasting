"""Corrected local eval — three Pearson r methods + population metrics.

Mirrors the math in scripts/eval_r_methods.py but reads the local cached
session arrays at data/processed/combined_steinmetz_ibl_cache/ instead of
downloading from S3. Writes per-session + aggregate metrics to JSON.

Usage (from repo root, with .venv activated):
    .venv/Scripts/python.exe scripts/eval_local_corrected.py \
        --ckpt outputs/s3_metrics/multihead-1l-v3/best_model.pt \
        --num-layers 1 \
        --output outputs/eval_local/multihead_1l_v3.json
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

from src.models.student import StudentSNN  # noqa: E402
from src.utils.device import resolve_device  # noqa: E402
from src.data.multi_session_loader import (  # noqa: E402
    MaskedSpikeCountDataset,
    pad_to_channels,
    build_channel_mask,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint .pt",
    )
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
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--neuron-type", type=str, default="rsynaptic")
    p.add_argument("--alpha", type=float, default=0.85)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--readout-mode", type=str, default="exponential")
    p.add_argument("--use-layer-norm", action="store_true", default=True)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learn-threshold", action="store_true", default=True)
    p.add_argument("--learn-beta", action="store_true", default=True)
    p.add_argument(
        "--auxiliary-heads",
        nargs="*",
        default=["stimulus", "response"],
    )
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON path",
    )
    p.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Limit sessions (debug).",
    )
    return p.parse_args()


def session_metrics(
    preds: torch.Tensor, targets: torch.Tensor
) -> dict:
    """Per-session metrics on real-neuron-only (T, M_i) tensors.

    Returns: dict with per_ch, weighted, global, pop_rate, spatial, cosine,
             per_neuron_r (list).
    """
    p = preds.double().numpy()
    t = targets.double().numpy()

    # Per-neuron r
    p_c = p - p.mean(axis=0, keepdims=True)
    t_c = t - t.mean(axis=0, keepdims=True)
    num = (p_c * t_c).sum(axis=0)
    den = np.sqrt((p_c ** 2).sum(axis=0) * (t_c ** 2).sum(axis=0))
    safe = den > 1e-12
    per_neuron_r = np.zeros_like(num)
    per_neuron_r[safe] = num[safe] / den[safe]

    per_ch = float(np.nanmean(per_neuron_r))

    # Activity-weighted: weight per-neuron r by total target spike count
    activity = t.sum(axis=0)
    activity_norm = activity / (activity.sum() + 1e-12)
    weighted = float((per_neuron_r * activity_norm).sum())

    # Global flatten r
    fp = p.reshape(-1)
    ft = t.reshape(-1)
    fp_c = fp - fp.mean()
    ft_c = ft - ft.mean()
    fnum = (fp_c * ft_c).sum()
    fden = np.sqrt((fp_c ** 2).sum() * (ft_c ** 2).sum())
    global_r = float(fnum / fden) if fden > 1e-12 else 0.0

    # Population rate over time
    pop_p = p.sum(axis=1)
    pop_t = t.sum(axis=1)
    if pop_p.std() > 0 and pop_t.std() > 0:
        pop_rate_r = float(np.corrcoef(pop_p, pop_t)[0, 1])
    else:
        pop_rate_r = 0.0

    # Spatial pattern (per-bin neuron-vector correlation)
    spat_corrs = []
    cos_vals = []
    for i in range(p.shape[0]):
        pi = p[i]
        ti = t[i]
        if pi.std() > 0 and ti.std() > 0:
            spat_corrs.append(float(np.corrcoef(pi, ti)[0, 1]))
        npi = np.linalg.norm(pi)
        nti = np.linalg.norm(ti)
        if npi > 0 and nti > 0:
            cos_vals.append(float(pi @ ti / (npi * nti)))
    spatial_r = float(np.mean(spat_corrs)) if spat_corrs else 0.0
    cosine_sim = float(np.mean(cos_vals)) if cos_vals else 0.0

    # MAE
    mae = float(np.abs(p - t).mean())

    return {
        "per_ch_r": per_ch,
        "weighted_r": weighted,
        "global_r": global_r,
        "pop_rate_r": pop_rate_r,
        "spatial_r": spatial_r,
        "cosine_sim": cosine_sim,
        "mae": mae,
        "per_neuron_r": per_neuron_r.tolist(),
    }


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

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)

    aux = args.auxiliary_heads if args.auxiliary_heads else None
    model = StudentSNN(
        input_size=m_max,
        output_size=m_max,
        hidden_size=args.hidden_size,
        beta=args.beta,
        threshold=args.threshold,
        gradient_slope=25.0,
        learn_beta=args.learn_beta,
        num_layers=args.num_layers,
        neuron_type=args.neuron_type,
        alpha=args.alpha,
        use_layer_norm=args.use_layer_norm,
        dropout=args.dropout,
        learn_threshold=args.learn_threshold,
        readout_mode=args.readout_mode,
        auxiliary_heads=aux,
    )
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        print(f"strict=True failed: {e}\nFalling back to strict=False")
        model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Loaded {args.ckpt}: epoch={ckpt.get('epoch', '?')}, "
        f"layers={args.num_layers}, params={n_params:,}"
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
            print(f"  Skip session {sess_idx:03d}: missing .npy")
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
            if isinstance(out, tuple):
                rates = out[0]
            elif isinstance(out, dict):
                rates = out["rates"]
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

    # ---- Aggregate ----
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

    def simple_mean(key: str) -> float:
        return float(
            np.mean([s[key] for s in per_session_results])
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
        "session_mean": {
            k: simple_mean(k)
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

    # Source breakdown
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
        "num_layers": args.num_layers,
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
    print(f"  AGGREGATE — {args.ckpt}")
    print("=" * 72)
    print(
        f"  Sessions: {aggregate['n_sessions']}, "
        f"neurons: {aggregate['total_neurons']:,}, "
        f"bins: {aggregate['total_test_bins']:,}"
    )
    nw = aggregate["neuron_weighted"]
    print(f"  Neuron-weighted across sessions:")
    print(
        f"    per_ch_r   = {nw['per_ch_r']:.4f}    "
        f"weighted_r = {nw['weighted_r']:.4f}    "
        f"global_r   = {nw['global_r']:.4f}"
    )
    print(
        f"    pop_rate_r = {nw['pop_rate_r']:.4f}    "
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
            f"weighted={snw['weighted_r']:.4f} "
            f"global={snw['global_r']:.4f} "
            f"pop={snw['pop_rate_r']:.4f} "
            f"cos={snw['cosine_sim']:.4f}"
        )
    print(f"  Wrote: {out_path}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
