"""F2 cross-arch decodability variants.

Produces multiple visual treatments of the same underlying data:
  - v1_forest_6panel: current default (forest plot, 6 panels)
  - v2_dumbbell_single: single-panel dumbbell, task x architecture
  - v3_grouped_bars: grouped bar chart, tasks grouped by architecture
  - v4_delta_from_raw: delta-from-raw-counts view (most provocative framing)
  - v5_compact_summary: 2-row compact version for space-constrained slots
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from figures.style import apply_style, save_figure, TEXT_WIDTH, COLORS


READOUTS = [
    (
        "linear_steinmetz",
        "Linear / raw counts",
        COLORS["Ground Truth"],
        "^",
    ),
    (
        "transformer",
        "Transformer",
        COLORS["Transformer"],
        "s",
    ),
    ("lru", "LRU", COLORS["LRU"], "o"),
    ("mamba", "Mamba", COLORS["Mamba"], "D"),
    (
        "snn_standalone_v12b",
        "Spiking NN",
        COLORS["SNN"],
        "P",
    ),
]


def load_all():
    results_dir = PROJECT_ROOT / "outputs" / "eval_local"
    data = {}
    for tag, *_ in READOUTS:
        p = results_dir / f"behavioral_decode_{tag}.json"
        if p.exists():
            data[tag] = json.load(open(p))
    return data


def v1_forest_6panel(data, out_dir):
    """6-panel grid: tasks x {bin, trial}."""
    apply_style()
    fig, axes = plt.subplots(
        2, 3, figsize=(TEXT_WIDTH, 3.4),
        constrained_layout=True, sharey=False,
    )
    n = len(READOUTS)
    y = np.arange(n)[::-1]
    tasks = [
        ("response_bin", "Response 3-class", 1 / 3),
        ("stimulus_bin_16class", "Stimulus 16-class", 1 / 16),
        ("stimulus_bin_side3", "Stim side 3-class", 1 / 3),
    ]
    trial = [
        ("resp_3_majority", 1 / 3),
        ("stim_16_majority", 1 / 16),
        ("side_3_majority", 1 / 3),
    ]
    for col, (key, title, chance) in enumerate(tasks):
        ax = axes[0, col]
        for i, (tag, label, color, mk) in enumerate(READOUTS):
            d = data.get(tag)
            if not d:
                continue
            m = d[key]
            ax.errorbar(
                m["acc"], y[i],
                xerr=[[m["acc"] - m["ci95_lo"]], [m["ci95_hi"] - m["acc"]]],
                fmt=mk, color=color, markersize=5,
                capsize=2.5, linewidth=0.9, markeredgewidth=0,
            )
        ax.axvline(chance, color="#bbb", ls=":", lw=0.6, zorder=0)
        ax.set_title(title + "\n(bin-level)", fontsize=8.5)
        ax.set_yticks(y)
        ax.set_yticklabels([r[1] for r in READOUTS] if col == 0 else [], fontsize=7.5)
        ax.tick_params(labelsize=7)
    for col, (key, chance) in enumerate(trial):
        ax = axes[1, col]
        title = tasks[col][1] + "\n(trial vote)"
        for i, (tag, label, color, mk) in enumerate(READOUTS):
            d = data.get(tag)
            if not d:
                continue
            v = d["trial_level"][key]
            ax.scatter(v, y[i], color=color, marker=mk, s=34, zorder=2, edgecolors="none")
        ax.axvline(chance, color="#bbb", ls=":", lw=0.6, zorder=0)
        ax.set_title(title, fontsize=8.5)
        ax.set_yticks(y)
        ax.set_yticklabels([r[1] for r in READOUTS] if col == 0 else [], fontsize=7.5)
        ax.set_xlabel("Accuracy", fontsize=8)
        ax.tick_params(labelsize=7)
    save_figure(fig, "v1_forest_6panel", out_dir=out_dir)
    plt.close(fig)


def v2_dumbbell_single(data, out_dir):
    """Single-panel dumbbell: trial-level response vs raw-count baseline."""
    apply_style()
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.7, 2.5), constrained_layout=True)
    # Compare trial-level response vote (main headline)
    baseline = data["linear_steinmetz"]["trial_level"]["resp_3_majority"]
    bar_y = []
    for i, (tag, label, color, mk) in enumerate(READOUTS):
        if tag == "linear_steinmetz":
            continue
        d = data.get(tag)
        v = d["trial_level"]["resp_3_majority"]
        y = len(bar_y)
        # line from baseline to value
        ax.plot(
            [baseline, v], [y, y],
            color=color, lw=2.0, alpha=0.7, zorder=1,
        )
        ax.scatter([baseline], [y], marker="|", color="#888", s=80, zorder=2)
        ax.scatter(
            [v], [y], marker=mk, color=color, s=60,
            zorder=3, edgecolors="white", linewidths=0.8,
        )
        bar_y.append((y, label, v - baseline, v))
    # Baseline reference line
    ax.axvline(
        baseline, color="#666", ls="--", lw=0.8, zorder=0,
        label=f"Raw counts = {baseline:.3f}",
    )
    ax.axvline(1 / 3, color="#ccc", ls=":", lw=0.6, zorder=0)
    ax.text(1 / 3, -0.7, "chance", fontsize=7, ha="center", color="#888")
    ax.set_yticks([b[0] for b in bar_y])
    ax.set_yticklabels([b[1] for b in bar_y], fontsize=8)
    ax.set_xlabel("Response 3-class trial-level accuracy", fontsize=8.5)
    ax.set_xlim(0.3, max(b[3] for b in bar_y) + 0.04)
    # Annotate deltas
    for y, label, delta, v in bar_y:
        sign = "+" if delta >= 0 else ""
        ax.annotate(
            f"{sign}{delta*100:.1f} pts",
            xy=(v, y), xytext=(6, 0), textcoords="offset points",
            fontsize=7, va="center",
            color="#333",
        )
    ax.set_title(
        "Decodability gap vs raw-count baseline "
        "(trial-level response vote)",
        fontsize=9, loc="left",
    )
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    save_figure(fig, "v2_dumbbell_single", out_dir=out_dir)
    plt.close(fig)


def v3_grouped_bars(data, out_dir):
    """Grouped bars: 3 tasks x N architectures, trial-level only."""
    apply_style()
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.85, 2.6), constrained_layout=True)

    keys = [
        ("resp_3_majority", "Response 3", 1 / 3),
        ("stim_16_majority", "Stimulus 16", 1 / 16),
        ("side_3_majority", "Stim side 3", 1 / 3),
    ]
    n_tasks = len(keys)
    n_arch = len(READOUTS)
    x = np.arange(n_tasks)
    width = 0.14
    for i, (tag, label, color, mk) in enumerate(READOUTS):
        d = data.get(tag)
        if not d:
            continue
        vals = [d["trial_level"][k[0]] for k in keys]
        offset = (i - (n_arch - 1) / 2) * width
        ax.bar(
            x + offset, vals, width,
            color=color, label=label,
            edgecolor="white", linewidth=0.5,
        )
    # chance lines per task
    for xi, (_, _, chance) in enumerate(keys):
        ax.hlines(
            chance,
            xmin=xi - n_arch * width / 2,
            xmax=xi + n_arch * width / 2,
            colors="#bbb", linestyles=":", lw=0.7, zorder=0,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([k[1] for k in keys], fontsize=9)
    ax.set_ylabel("Trial-level accuracy", fontsize=9)
    ax.set_ylim(0, 0.95)
    ax.legend(
        ncol=2, fontsize=7, loc="upper right",
        columnspacing=1.0, handletextpad=0.4,
        frameon=False, bbox_to_anchor=(1.0, 1.02),
    )
    ax.set_title(
        "Behavioral decodability by feature source",
        fontsize=9, loc="left",
    )
    save_figure(fig, "v3_grouped_bars", out_dir=out_dir)
    plt.close(fig)


def v4_delta_from_raw(data, out_dir):
    """Delta vs raw-count ceiling (the provocative framing)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.5), constrained_layout=True)

    baseline = data["linear_steinmetz"]
    # Use trial-level metrics; compute delta
    keys = [
        ("resp_3_majority", "Response 3"),
        ("stim_16_majority", "Stim 16"),
        ("side_3_majority", "Stim side 3"),
    ]
    arch_order = [r for r in READOUTS if r[0] != "linear_steinmetz"]
    n_arch = len(arch_order)
    n_k = len(keys)
    x = np.arange(n_arch)
    width = 0.25
    for j, (kkey, klabel) in enumerate(keys):
        base = baseline["trial_level"][kkey]
        vals = [
            data[tag]["trial_level"][kkey] - base for tag, *_ in arch_order
        ]
        offset = (j - (n_k - 1) / 2) * width
        colors = [
            COLORS["SNN"] if tag.startswith("snn") else (
                COLORS["Transformer"] if tag == "transformer"
                else COLORS["LRU"] if tag == "lru"
                else COLORS["Mamba"]
            )
            for tag, *_ in arch_order
        ]
        # Single color per task
        task_colors = ["#0072B2", "#D55E00", "#CC79A7"]
        ax.bar(
            x + offset, vals, width,
            color=task_colors[j],
            label=klabel,
            edgecolor="white", linewidth=0.5,
            alpha=0.9,
        )
    ax.axhline(0, color="#444", lw=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([r[1] for r in arch_order], fontsize=8.5)
    ax.set_ylabel(
        "\u0394 trial-accuracy vs linear/raw counts",
        fontsize=9,
    )
    ax.text(
        -0.5, 0.002, "better than raw",
        fontsize=7, color="#009E73", va="bottom",
    )
    ax.text(
        -0.5, -0.002, "worse than raw",
        fontsize=7, color="#D55E00", va="top",
    )
    ax.legend(
        ncol=3, fontsize=7.5, loc="upper center",
        frameon=False, bbox_to_anchor=(0.5, 1.08),
    )
    ax.set_title(
        "The neuromorphic decoding tax "
        "(\u0394 from linear-on-raw-counts ceiling)",
        fontsize=9.5, loc="left", pad=14,
    )
    save_figure(fig, "v4_delta_from_raw", out_dir=out_dir)
    plt.close(fig)


def v5_compact_summary(data, out_dir):
    """Two-column compact version for tight main-text slots."""
    apply_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH, 2.2),
        constrained_layout=True,
    )
    for col, (metric_key, title, chance) in enumerate([
        ("resp_3_majority", "Response 3-class (trial vote)", 1 / 3),
        ("stim_16_majority", "Stimulus 16-class (trial vote)", 1 / 16),
    ]):
        ax = axes[col]
        for i, (tag, label, color, mk) in enumerate(READOUTS):
            d = data.get(tag)
            if not d:
                continue
            v = d["trial_level"][metric_key]
            ax.scatter(
                v, i, marker=mk, color=color, s=90,
                edgecolors="white", linewidths=1.2, zorder=3,
            )
        ax.axvline(chance, color="#bbb", ls=":", lw=0.6)
        ax.set_yticks(range(len(READOUTS)))
        if col == 0:
            ax.set_yticklabels([r[1] for r in READOUTS], fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Accuracy", fontsize=8.5)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(
            min(chance - 0.05, 0.2),
            max(0.85, max(d["trial_level"][metric_key] for d in data.values()) + 0.05)
        )
        ax.invert_yaxis()
    save_figure(fig, "v5_compact_summary", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    data = load_all()
    out_dir = (
        PROJECT_ROOT / "docs" / "neurips_neurocog"
        / "figure_candidates" / "F2_cross_arch"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(data)} readouts: {list(data.keys())}")
    for fn in [
        v1_forest_6panel,
        v2_dumbbell_single,
        v3_grouped_bars,
        v4_delta_from_raw,
        v5_compact_summary,
    ]:
        print(f"Generating {fn.__name__}...")
        fn(data, out_dir)
    print(f"Wrote {len(list(out_dir.glob('*.png')))} PNGs to {out_dir}")
