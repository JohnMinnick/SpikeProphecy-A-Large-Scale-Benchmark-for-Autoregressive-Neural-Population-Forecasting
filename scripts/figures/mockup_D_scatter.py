"""Hero figure mockup D: scatter of forecaster decoding vs. raw-counts decoding.

Each point = one session. Y = decoding accuracy from forecaster predictions.
X = decoding accuracy from raw spike counts on same session/trials. Color
by architecture. Diagonal = identity. Above-diagonal = denoising; below =
decoding tax.

Plus marginal kernel-density of (y - x) on the right.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from figures.style import apply_style, save_figure, TEXT_WIDTH, COLORS


def load_per_session(tag, key="resp_acc"):
    p = PROJECT_ROOT / "outputs" / "eval_local" / f"behavioral_decode_{tag}.json"
    if not p.exists():
        return {}
    d = json.load(open(p))
    return {
        s["session_idx"]: s[key]
        for s in d.get("per_session", [])
        if key in s
    }


def main():
    apply_style()

    # Reference: linear-on-raw-counts per-session
    raw = load_per_session("linear_steinmetz", key="resp_acc")
    arch_tags = [
        ("transformer", "Transformer", COLORS["Transformer"]),
        ("mamba", "Mamba", COLORS["Mamba"]),
        ("lru", "LRU", COLORS["LRU"]),
        ("snn_standalone_v12b", "Spiking NN", COLORS["SNN"]),
    ]

    fig = plt.figure(figsize=(TEXT_WIDTH, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.0], wspace=0.10)

    # --- Main scatter ---
    ax = fig.add_subplot(gs[0])
    # Diagonal identity
    lo, hi = 0.10, 0.85
    ax.plot([lo, hi], [lo, hi], color="#444444", linewidth=1.0,
            linestyle="--", zorder=1, label="_diagonal")
    # Shade above and below
    ax.fill_between(
        [lo, hi], [lo, hi], [hi, hi],
        color="#1a6d2a", alpha=0.06, zorder=0,
    )
    ax.fill_between(
        [lo, hi], [lo, lo], [lo, hi],
        color="#a02929", alpha=0.06, zorder=0,
    )
    # Chance lines
    ax.axhline(1 / 3, color="#bbbbbb", linewidth=0.5, linestyle=":", zorder=0)
    ax.axvline(1 / 3, color="#bbbbbb", linewidth=0.5, linestyle=":", zorder=0)

    diff_by_arch = {}
    for tag, label, color in arch_tags:
        per_sess = load_per_session(tag, key="resp_acc")
        x = []
        y = []
        for sidx, raw_acc in raw.items():
            if sidx in per_sess:
                x.append(raw_acc)
                y.append(per_sess[sidx])
        x = np.array(x)
        y = np.array(y)
        diff_by_arch[label] = (color, y - x)
        ax.scatter(
            x, y, color=color, s=18, alpha=0.65, edgecolors="none",
            label=f"{label}  (Δ̄ = {np.mean(y - x)*100:+.1f} pts)",
            zorder=3,
        )

    # Annotations: above/below
    ax.text(
        0.78, 0.80, "denoising",
        fontsize=9.5, color="#1a6d2a", style="italic", fontweight="bold",
        ha="right", va="top",
    )
    ax.text(
        0.78, 0.74, "(above diagonal)",
        fontsize=7, color="#1a6d2a", style="italic",
        ha="right", va="top",
    )
    ax.text(
        0.78, 0.18, "decoding tax",
        fontsize=9.5, color="#a02929", style="italic", fontweight="bold",
        ha="right", va="bottom",
    )
    ax.text(
        0.78, 0.13, "(below diagonal)",
        fontsize=7, color="#a02929", style="italic",
        ha="right", va="bottom",
    )

    ax.set_xlabel(
        "Per-session decoding from raw spike counts", fontsize=8.5,
    )
    ax.set_ylabel(
        "Per-session decoding from forecaster predictions", fontsize=8.5,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(
        loc="lower right", fontsize=7.5, frameon=True,
        framealpha=0.85, edgecolor="#cccccc",
    )
    ax.set_title(
        "Each point = 1 session × architecture; metric = bin-level "
        "response 3-class accuracy (held-out trials)",
        fontsize=8.5, loc="left",
    )
    ax.text(
        -0.12, 1.02, "a", transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="top",
    )

    # --- Right: marginal Δ density per architecture ---
    ax2 = fig.add_subplot(gs[1])
    n_archs = len(arch_tags)
    # Use a rotated histogram-style: KDE for each arch
    from scipy.stats import gaussian_kde

    delta_range = np.linspace(-0.15, 0.15, 200)
    for i, (label, (color, diffs)) in enumerate(diff_by_arch.items()):
        if len(diffs) > 1:
            try:
                kde = gaussian_kde(diffs, bw_method=0.5)
                d = kde(delta_range)
                # Plot horizontally
                ax2.fill_betweenx(
                    delta_range, 0, d, color=color, alpha=0.5,
                    label=label,
                )
                ax2.plot(d, delta_range, color=color, linewidth=0.8)
            except Exception:
                pass

    ax2.axhline(0, color="#444444", linewidth=1.0, linestyle="--", zorder=1)
    ax2.set_ylim(-0.15, 0.15)
    ax2.set_xticks([])
    ax2.set_ylabel(
        "Δ accuracy (pred − raw)", fontsize=8.5,
    )
    ax2.set_xlabel("density", fontsize=7.5)
    ax2.text(
        -0.20, 1.02, "b", transform=ax2.transAxes,
        fontsize=12, fontweight="bold", va="top",
    )
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    fig.suptitle(
        "Forecaster architecture, not session, drives the decoding asymmetry",
        fontsize=11, fontweight="bold", y=1.02,
    )

    out_dir = PROJECT_ROOT / "docs" / "neurips_neurocog" / "figures" / "mockups"
    save_figure(fig, "hero_D_scatter", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
