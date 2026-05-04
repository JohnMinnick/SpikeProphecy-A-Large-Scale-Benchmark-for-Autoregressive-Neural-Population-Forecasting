"""F1 Hero rebuilt for the unified forecast-and-decode framing.

Variants:
  v5_unified_pipeline: 2-panel hero — (a) schematic of one-model-two-outputs,
      (b) the headline 'Mamba beats raw counts on trial-vote decoding'
  v6_pipeline_and_tradeoff: 3-panel — pipeline + capability + neuromorphic tradeoff
  v7_capability_front_and_center: single provocative panel with the
      'one model, two outputs' claim + quant overlay
  v4 (existing) relabeled to match new framing
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from figures.style import apply_style, save_figure, TEXT_WIDTH, COLORS


def _load():
    d = PROJECT_ROOT / "outputs" / "eval_local"
    tags = ["linear_steinmetz", "mamba", "transformer", "lru",
            "snn_standalone_v12b"]
    return {t: json.load(open(d / f"behavioral_decode_{t}.json"))
            for t in tags
            if (d / f"behavioral_decode_{t}.json").exists()}


def _box(ax, x, y, w, h, label, color="#222", facecolor="white",
         fontsize=8, fontweight="bold"):
    b = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025", linewidth=1.0,
        edgecolor=color, facecolor=facecolor,
    )
    ax.add_patch(b)
    ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center",
        fontsize=fontsize, color=color, fontweight=fontweight,
    )


def _arrow(ax, x1, y1, x2, y2, color="#888", lw=1.0, style="->"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color,
        linewidth=lw, mutation_scale=12,
        shrinkA=5, shrinkB=5,
    )
    ax.add_patch(a)


def v5_unified_pipeline(data, out_dir):
    """2-panel hero: schematic of unified pipeline + headline bar chart."""
    apply_style()
    fig = plt.figure(figsize=(TEXT_WIDTH, 3.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])

    # ---- Panel a: pipeline schematic ----
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 5.5)
    ax_a.axis("off")
    ax_a.text(
        0.02, 5.3, "a", fontsize=13, fontweight="bold",
        va="top", color="#222",
    )

    # Input
    _box(ax_a, 0.2, 2.3, 1.9, 1.2, "Spike counts\n(M neurons, 10 bins)",
         color="#444", fontsize=7.5)

    # Single model — big central box
    _box(ax_a, 3.2, 2.1, 2.3, 1.6, "Mamba\nforecaster\n(one model)",
         color=COLORS["Mamba"], facecolor="#fdf4ee", fontsize=9)
    _arrow(ax_a, 2.1, 2.9, 3.2, 2.9, color="#888")

    # Two outputs
    _box(ax_a, 6.3, 3.5, 2.1, 1.1, "Predicted\nrates  \u03BB(t+1)",
         color="#0072B2", facecolor="#f0f6fa", fontsize=8)
    _box(ax_a, 6.3, 0.8, 2.1, 1.1, "Linear head\n(per session)",
         color="#009E73", facecolor="#f0f8f4", fontsize=8)
    _arrow(ax_a, 5.5, 3.1, 6.3, 4.05, color="#aaa")
    _arrow(ax_a, 5.5, 2.7, 6.3, 1.35, color="#aaa")

    # Behavior readout box
    _box(ax_a, 8.6, 0.8, 1.4, 1.1,
         "Choice\nStim side",
         color="#222", facecolor="white", fontsize=8)
    _arrow(ax_a, 8.4, 1.35, 8.6, 1.35, color="#aaa")

    # Labels
    ax_a.text(
        7.35, 4.8, "population forecast",
        fontsize=7.5, color="#0072B2", ha="center", style="italic",
    )
    ax_a.text(
        7.35, 0.35, "behavioral readout",
        fontsize=7.5, color="#009E73", ha="center", style="italic",
    )
    ax_a.text(
        5.0, 4.6,
        "one model  \u2192  one forward pass  \u2192  two useful outputs",
        fontsize=8, ha="center", color="#333", fontweight="bold",
    )

    # ---- Panel b: headline capability ----
    ax_b = fig.add_subplot(gs[1])
    ax_b.text(
        -0.15, 1.02, "b", fontsize=13, fontweight="bold",
        va="top", transform=ax_b.transAxes, color="#222",
    )
    readouts = [
        ("linear_steinmetz", "Linear /\nraw counts", COLORS["Ground Truth"]),
        ("mamba", "Mamba\nforecast", COLORS["Mamba"]),
    ]
    keys = [
        ("resp_3_majority", "Response (3-class)", 1 / 3),
        ("side_3_majority", "Stim side (3-class)", 1 / 3),
    ]
    n_k = len(keys)
    n_r = len(readouts)
    x = np.arange(n_k)
    w = 0.32
    for i, (tag, lbl, color) in enumerate(readouts):
        d = data[tag]
        vals = [d["trial_level"][k[0]] for k in keys]
        offset = (i - (n_r - 1) / 2) * w
        ax_b.bar(
            x + offset, vals, w, color=color, alpha=0.9,
            edgecolor="white", linewidth=0.6, label=lbl,
        )
        for xi, val in enumerate(vals):
            ax_b.text(
                x[xi] + offset, val + 0.015, f"{val*100:.0f}%",
                fontsize=7.5, ha="center", va="bottom",
                color=color, fontweight="bold",
            )
    # Delta annotations
    for xi, (k, _, chance) in enumerate(keys):
        raw = data["linear_steinmetz"]["trial_level"][k]
        mamba = data["mamba"]["trial_level"][k]
        delta = (mamba - raw) * 100
        sign = "+" if delta >= 0 else ""
        ax_b.annotate(
            f"{sign}{delta:.0f} pp", xy=(xi, mamba), xytext=(0, 14),
            textcoords="offset points", fontsize=7.5, ha="center",
            color="#009E73", fontweight="bold",
        )
    for xi, (_, _, chance) in enumerate(keys):
        ax_b.hlines(
            chance, xmin=xi - 0.4, xmax=xi + 0.4,
            colors="#bbb", linestyles=":", lw=0.7, zorder=0,
        )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([k[1] for k in keys], fontsize=8)
    ax_b.set_ylabel("Trial-level accuracy", fontsize=9)
    ax_b.set_ylim(0, 1.0)
    ax_b.legend(
        loc="upper center", fontsize=7.5, frameon=False,
        ncol=2, bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.2,
    )
    ax_b.set_title(
        "Forecast predictions beat raw counts\nat behavioral decoding",
        fontsize=9, loc="left", pad=22,
    )

    save_figure(fig, "v5_unified_pipeline", out_dir=out_dir)
    plt.close(fig)


def v6_pipeline_and_tradeoff(data, out_dir):
    """3-panel: pipeline + capability + neuromorphic trade-off."""
    apply_style()
    fig = plt.figure(figsize=(TEXT_WIDTH, 3.6), constrained_layout=True)
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.3, 1.0, 1.0],
    )

    # Panel a: compact schematic
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 5.5)
    ax_a.axis("off")
    ax_a.text(0.02, 5.3, "a", fontsize=13, fontweight="bold", color="#222")
    _box(ax_a, 0.2, 2.3, 1.8, 1.2, "Spike\ncounts", "#444", fontsize=7.5)
    _box(ax_a, 3.0, 2.1, 2.2, 1.6, "One\nforecaster", COLORS["Mamba"],
         "#fdf4ee", fontsize=9)
    _box(ax_a, 6.2, 3.5, 2.0, 1.1, "Rates", "#0072B2", "#f0f6fa",
         fontsize=8)
    _box(ax_a, 6.2, 0.8, 2.0, 1.1, "Linear\nhead", "#009E73", "#f0f8f4",
         fontsize=8)
    _box(ax_a, 8.5, 0.8, 1.4, 1.1, "Choice\nStim", "#222", "white",
         fontsize=7.5)
    _arrow(ax_a, 2.0, 2.9, 3.0, 2.9)
    _arrow(ax_a, 5.2, 3.1, 6.2, 4.05)
    _arrow(ax_a, 5.2, 2.7, 6.2, 1.35)
    _arrow(ax_a, 8.2, 1.35, 8.5, 1.35)
    ax_a.text(
        5.0, 4.8, "one model, two outputs",
        fontsize=8.5, ha="center", color="#333", fontweight="bold",
    )

    # Panel b: capability (Mamba vs raw counts)
    ax_b = fig.add_subplot(gs[1])
    ax_b.text(-0.15, 1.02, "b", fontsize=13, fontweight="bold",
              transform=ax_b.transAxes, color="#222")
    keys = [
        ("resp_3_majority", "Response", 1 / 3),
        ("side_3_majority", "Stim side", 1 / 3),
    ]
    x = np.arange(len(keys))
    w = 0.35
    ax_b.bar(
        x - w / 2,
        [data["linear_steinmetz"]["trial_level"][k[0]] for k in keys],
        w, color=COLORS["Ground Truth"], alpha=0.85,
        label="Raw counts", edgecolor="white", linewidth=0.5,
    )
    ax_b.bar(
        x + w / 2,
        [data["mamba"]["trial_level"][k[0]] for k in keys],
        w, color=COLORS["Mamba"], alpha=0.95,
        label="Mamba forecast", edgecolor="white", linewidth=0.5,
    )
    for xi, (k, _, chance) in enumerate(keys):
        ax_b.hlines(
            chance, xmin=xi - 0.4, xmax=xi + 0.4,
            colors="#bbb", linestyles=":", lw=0.6,
        )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([k[1] for k in keys], fontsize=8)
    ax_b.set_ylabel("Trial-vote accuracy", fontsize=8.5)
    ax_b.set_ylim(0, 0.92)
    ax_b.legend(fontsize=7, frameon=False, loc="upper center", ncol=2)
    ax_b.set_title("Capability", fontsize=9, loc="left", pad=18)

    # Panel c: SNN vs Mamba tradeoff
    ax_c = fig.add_subplot(gs[2])
    ax_c.text(-0.15, 1.02, "c", fontsize=13, fontweight="bold",
              transform=ax_c.transAxes, color="#222")
    tags = [
        ("mamba", "Mamba\n(2.3M)", COLORS["Mamba"]),
        ("snn_standalone_v12b", "Spiking NN\n(0.7M)", COLORS["SNN"]),
    ]
    xs = np.arange(len(tags))
    cos = {
        "mamba": 0.640,
        "snn_standalone_v12b": 0.494,
    }
    ax_c.bar(
        xs - 0.2,
        [cos[t[0]] for t in tags],
        0.38, color=[t[2] for t in tags], alpha=0.85,
        label="Cosine fidelity",
        edgecolor="white", linewidth=0.5,
    )
    ax_c.bar(
        xs + 0.2,
        [data[t[0]]["trial_level"]["resp_3_majority"] for t in tags],
        0.38, color=[t[2] for t in tags], alpha=0.5,
        label="Response trial-vote",
        edgecolor="white", linewidth=0.5,
    )
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels([t[1] for t in tags], fontsize=8)
    ax_c.set_ylim(0, 0.85)
    ax_c.set_ylabel("Score", fontsize=8.5)
    ax_c.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax_c.set_title("Neuromorphic variant", fontsize=9, loc="left", pad=18)

    save_figure(fig, "v6_pipeline_and_tradeoff", out_dir=out_dir)
    plt.close(fig)


def v7_capability_front_and_center(data, out_dir):
    """Single-panel: the unified-capability claim with 4 bars on one axis."""
    apply_style()
    fig, ax = plt.subplots(
        figsize=(TEXT_WIDTH * 0.85, 2.8), constrained_layout=True,
    )
    # Show: raw counts, Mamba, Transformer, LRU, SNN
    rows = [
        ("linear_steinmetz", "Linear / raw counts\n(no forecaster)",
         COLORS["Ground Truth"]),
        ("mamba", "Mamba forecast\n+ linear readout",
         COLORS["Mamba"]),
        ("transformer", "Transformer forecast\n+ linear readout",
         COLORS["Transformer"]),
        ("lru", "LRU forecast\n+ linear readout",
         COLORS["LRU"]),
        ("snn_standalone_v12b", "Spiking NN forecast\n+ linear readout",
         COLORS["SNN"]),
    ]
    y = np.arange(len(rows))
    resp = [data[t[0]]["trial_level"]["resp_3_majority"] for t in rows]
    side = [data[t[0]]["trial_level"]["side_3_majority"] for t in rows]
    w = 0.35
    ax.barh(
        y - w / 2, resp, w,
        color=[t[2] for t in rows], alpha=0.95,
        label="Response (3-class, chance = 33%)",
        edgecolor="white", linewidth=0.5,
    )
    ax.barh(
        y + w / 2, side, w,
        color=[t[2] for t in rows], alpha=0.55,
        label="Stim side (3-class, chance = 33%)",
        edgecolor="white", linewidth=0.5,
    )
    ax.axvline(1 / 3, color="#bbb", ls=":", lw=0.6, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in rows], fontsize=8)
    ax.set_xlabel("Trial-level accuracy", fontsize=9)
    ax.set_xlim(0, 0.92)
    ax.invert_yaxis()
    ax.legend(
        fontsize=7, frameon=False, loc="lower right", ncol=1,
    )
    ax.set_title(
        "Any sequence forecaster's predictions decode behavior as well as "
        "or better than raw counts\u2014except the spiking variant, which "
        "pays a measurable tax.",
        fontsize=8.5, loc="left", wrap=True,
    )
    save_figure(fig, "v7_capability_front_and_center", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    data = _load()
    out_dir = (
        PROJECT_ROOT / "docs" / "neurips_neurocog"
        / "figure_candidates" / "F1_hero"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for fn in [v5_unified_pipeline, v6_pipeline_and_tradeoff,
               v7_capability_front_and_center]:
        print(f"Generating {fn.__name__}...")
        fn(data, out_dir)
    print(f"Wrote candidates to {out_dir}")
