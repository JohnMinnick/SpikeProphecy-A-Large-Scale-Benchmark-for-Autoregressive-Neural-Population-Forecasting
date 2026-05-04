"""F4 region hierarchy variants.

  v1_dumbbell_current: existing top-20 dumbbell (1L + 2L SNN)
  v2_grouped_by_system: top 20 grouped + colored by functional system
  v3_functional_summary: 8 functional-system aggregates, small bar chart
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from figures.style import apply_style, save_figure, TEXT_WIDTH, COLORS
from figure_region_hierarchy import REGION_GROUP, GROUP_COLOR


def _load_stats():
    p = PROJECT_ROOT / "outputs" / "eval_local" / "region_hierarchy_stats.json"
    if not p.exists():
        return None
    return json.load(open(p))


def v1_dumbbell_current(out_dir):
    """Copy the existing figure into the candidates folder."""
    import shutil
    src = PROJECT_ROOT / "docs" / "neurips_neurocog" / "figures" / "figure_region_hierarchy.png"
    if src.exists():
        shutil.copy(src, out_dir / "v1_dumbbell_current.png")
        shutil.copy(src.with_suffix(".pdf"), out_dir / "v1_dumbbell_current.pdf")
        print("  v1: copied existing dumbbell figure")


def v2_grouped_by_system(stats, out_dir):
    """Horizontal bars, top 20, colored by functional system group."""
    apply_style()
    per_region = stats["per_region"].get("1L SNN") or list(stats["per_region"].values())[0]
    sorted_r = sorted(per_region.items(), key=lambda x: -x[1]["mean"])
    top = sorted_r[:20]

    fig, ax = plt.subplots(
        figsize=(TEXT_WIDTH * 0.75, 3.6),
        constrained_layout=True,
    )
    y = np.arange(len(top))[::-1]
    for i, (region, s) in enumerate(top):
        color = GROUP_COLOR.get(REGION_GROUP.get(region, "Other"), "#bbb")
        ax.barh(
            y[i], s["mean"],
            xerr=s["se"], color=color, alpha=0.9,
            error_kw={"lw": 0.6, "capsize": 2},
            height=0.7, edgecolor="white", linewidth=0.4,
        )
        ax.text(
            s["mean"] + s["se"] + 0.003, y[i],
            f"n={s['n']}", fontsize=6.5, va="center", color="#555",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in top], fontsize=8)
    ax.set_xlabel("Per-neuron Pearson $r$", fontsize=9)
    ax.set_title(
        "Top-20 brain regions by predictability (1L SNN)",
        fontsize=9.5, loc="left",
    )
    # Legend of functional systems present in the top 20
    groups_present = {
        REGION_GROUP.get(r[0], "Other") for r in top
    }
    from matplotlib.patches import Patch
    handles = [
        Patch(color=GROUP_COLOR[g], label=g)
        for g in sorted(groups_present)
    ]
    ax.legend(
        handles=handles, loc="lower right", fontsize=7,
        frameon=False, ncol=2, handlelength=1.0,
    )
    kw = stats["kruskal_wallis"].get("1L SNN", {})
    if kw:
        ax.text(
            0.98, 0.02,
            f"$H{{=}}{kw.get('H',0):.0f}$, $p{{<}}10^{{-300}}$",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="#333",
        )
    save_figure(fig, "v2_grouped_by_system", out_dir=out_dir)
    plt.close(fig)


def v3_functional_summary(stats, out_dir):
    """Aggregate per functional system (8 coarse groups)."""
    apply_style()
    per_region = stats["per_region"].get("1L SNN") or list(stats["per_region"].values())[0]
    # Aggregate means weighted by n
    agg = defaultdict(lambda: {"n": 0, "sum_mean_n": 0})
    for region, s in per_region.items():
        g = REGION_GROUP.get(region, "Other")
        agg[g]["n"] += s["n"]
        agg[g]["sum_mean_n"] += s["mean"] * s["n"]
    rows = []
    for g, v in agg.items():
        if v["n"] < 50:
            continue
        rows.append((g, v["sum_mean_n"] / max(v["n"], 1), v["n"]))
    rows.sort(key=lambda x: -x[1])

    fig, ax = plt.subplots(
        figsize=(TEXT_WIDTH * 0.7, 2.8),
        constrained_layout=True,
    )
    y = np.arange(len(rows))[::-1]
    colors = [GROUP_COLOR.get(r[0], "#bbb") for r in rows]
    ax.barh(
        y, [r[1] for r in rows], color=colors, alpha=0.95,
        height=0.7, edgecolor="white", linewidth=0.4,
    )
    for i, (g, m, n) in enumerate(rows):
        ax.text(
            m + 0.003, y[i], f"n={n:,}", fontsize=7,
            va="center", color="#555",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel("Mean per-neuron $r$ (neuron-weighted)", fontsize=9)
    ax.set_title(
        "Region predictability by functional system",
        fontsize=9.5, loc="left",
    )
    save_figure(fig, "v3_functional_summary", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = (
        PROJECT_ROOT / "docs" / "neurips_neurocog"
        / "figure_candidates" / "F4_region"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = _load_stats()
    v1_dumbbell_current(out_dir)
    if stats:
        v2_grouped_by_system(stats, out_dir)
        v3_functional_summary(stats, out_dir)
    print(f"Wrote {len(list(out_dir.glob('*.png')))} PNGs to {out_dir}")
