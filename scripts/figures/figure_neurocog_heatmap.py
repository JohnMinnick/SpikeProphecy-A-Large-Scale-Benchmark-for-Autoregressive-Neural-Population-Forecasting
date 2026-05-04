"""
Neuro&Cog paper Figure: 3-panel heatmap (GT vs Mamba vs SNN).

Renders the figure referenced by docs/neurips_neurocog/main.tex:369.
Uses cached full-inference arrays in outputs/full_inference/session_NNN.npz
(keys: gt, mamba_rates, snn_rates, m_actual). For visual fairness with the
discrete-count GT panel, model rates are Poisson-sampled before display.

Outputs PNG + PDF to docs/neurips_neurocog/figures/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from figures.style import apply_style, save_figure, TEXT_WIDTH


def per_neuron_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-column (neuron) Pearson r between (T, M) arrays."""
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    num = (a * b).sum(axis=0)
    den = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0)) + 1e-12
    r = num / den
    r[~np.isfinite(r)] = 0.0
    return r


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inference-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "full_inference"),
        help="Directory of session_NNN.npz cached inference arrays.",
    )
    p.add_argument(
        "--session",
        type=int,
        default=10,
        help="Steinmetz session index. Default 10 — close to median fidelity.",
    )
    p.add_argument(
        "--window-bins",
        type=int,
        default=300,
        help="Time window (bins) to display. 300 * 50ms = 15s.",
    )
    p.add_argument(
        "--start-bin",
        type=int,
        default=None,
        help="Start bin (default: middle of available window).",
    )
    p.add_argument(
        "--max-neurons",
        type=int,
        default=200,
        help="Show top-N neurons by mean rate (capped for legibility).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "docs" / "neurips_neurocog" / "figures"),
    )
    p.add_argument(
        "--name",
        type=str,
        default="figure1_twin_heatmap",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    inf_dir = Path(args.inference_dir)
    npz_path = inf_dir / f"session_{args.session:03d}.npz"
    if not npz_path.exists():
        raise SystemExit(f"Missing inference cache: {npz_path}")

    data = np.load(npz_path)
    m_act = int(data["m_actual"])
    gt = data["gt"][:, :m_act].astype(np.float32)
    mamba = data["mamba_rates"][:, :m_act].astype(np.float32)
    snn = data["snn_rates"][:, :m_act].astype(np.float32)
    T = gt.shape[0]

    # ---- Choose temporal window ----
    win = min(args.window_bins, T)
    if args.start_bin is None:
        start = max(0, (T - win) // 2)
    else:
        start = max(0, min(args.start_bin, T - win))
    end = start + win

    gt_w = gt[start:end]
    mamba_w = mamba[start:end]
    snn_w = snn[start:end]

    # ---- Sort neurons by mean rate (descending), trim to top-N ----
    mean_rate = gt_w.mean(axis=0)
    order = np.argsort(-mean_rate)
    keep = order[: args.max_neurons]
    gt_w = gt_w[:, keep]
    mamba_w = mamba_w[:, keep]
    snn_w = snn_w[:, keep]

    # ---- Compute per-neuron r over the FULL session, ALL neurons ----
    # (Avoid top-N bias: mean over the full neuron set is what the paper
    # reports at the population level. Numbers will still differ from
    # Tab:main-results which averages across all 105 sessions.)
    r_mamba = float(per_neuron_pearson(gt, mamba).mean())
    r_snn = float(per_neuron_pearson(gt, snn).mean())

    # ---- Poisson-sample predictions for visual parity with discrete GT ----
    mamba_samp = rng.poisson(np.clip(mamba_w, 0, 50)).astype(np.float32)
    snn_samp = rng.poisson(np.clip(snn_w, 0, 50)).astype(np.float32)

    # ---- Color scale from GT 99th percentile ----
    vmax = float(max(np.percentile(gt_w, 99), 1.0))

    # ---- Plot ----
    apply_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(TEXT_WIDTH, 2.2), sharey=True, constrained_layout=True
    )

    cmap = "magma"
    panels = [
        (gt_w, "Ground truth", None),
        (mamba_samp, f"Mamba teacher  ($r$ = {r_mamba:.3f})", None),
        (snn_samp, f"1L SNN twin  ($r$ = {r_snn:.3f})", None),
    ]
    for ax, (mat, title, _) in zip(axes, panels):
        ax.imshow(
            mat.T,
            aspect="auto",
            origin="upper",
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Time (50 ms bins)", fontsize=8)

    axes[0].set_ylabel("Neuron (sorted by rate)", fontsize=8)

    # Single shared colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax)
    )
    cbar = fig.colorbar(
        sm, ax=axes, fraction=0.025, pad=0.015, shrink=0.85, aspect=18
    )
    cbar.set_label("Spikes / bin", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    out_dir = Path(args.out_dir)
    save_figure(fig, args.name, out_dir=out_dir)
    plt.close(fig)

    # ---- Report ----
    print(
        f"Session {args.session}: M={m_act}, window=[{start}, {end}) "
        f"({win * 0.05:.1f}s), top-{args.max_neurons} neurons"
    )
    print(f"  Mamba per-neuron r (full session, top-N): {r_mamba:.4f}")
    print(f"  SNN   per-neuron r (full session, top-N): {r_snn:.4f}")


if __name__ == "__main__":
    main()
