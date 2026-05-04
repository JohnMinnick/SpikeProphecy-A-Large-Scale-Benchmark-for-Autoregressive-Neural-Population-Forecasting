"""Region hierarchy variant: sorted bar chart with functional groups colored.

Bars sorted by mean per-neuron r (1L SNN). Bar color = functional group
(Cortex/Thalamus/Hippocampus/Midbrain etc). Adds a small rank arrow
showing 1L vs 2L agreement.
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

from figures.style import apply_style, save_figure, TEXT_WIDTH


REGION_GROUP = {
    # (Subset; reused from figure_region_hierarchy.py)
    "MOs": "Cortex", "MOp": "Cortex", "VISp": "Cortex",
    "VISa": "Cortex", "VISl": "Cortex", "VISpm": "Cortex",
    "RSP": "Cortex", "ACA": "Cortex", "PL": "Cortex",
    "SSp": "Cortex", "SSs": "Cortex", "ILA": "Cortex",
    "ORB": "Cortex", "AUD": "Cortex", "ECT": "Cortex",
    "VISa": "Cortex", "VISam": "Cortex",
    "CA1": "HPF", "CA2": "HPF", "CA3": "HPF", "CA": "HPF",
    "DG": "HPF", "SUB": "HPF", "POST": "HPF", "PAR": "HPF",
    "CP": "Striatum", "ACB": "Striatum",
    "GPi": "Pallidum", "GPe": "Pallidum",
    "LP": "Thalamus", "MD": "Thalamus", "VPM": "Thalamus",
    "VPL": "Thalamus", "PO": "Thalamus", "VAL": "Thalamus",
    "RT": "Thalamus", "LD": "Thalamus", "LGd": "Thalamus",
    "MG": "Thalamus", "AV": "Thalamus", "AM": "Thalamus",
    "AD": "Thalamus", "CL": "Thalamus", "PCN": "Thalamus",
    "PF": "Thalamus", "CM": "Thalamus", "MH": "Thalamus",
    "LH": "Thalamus",
    "ZI": "Hypothalamus",
    "SCm": "Midbrain", "SCs": "Midbrain", "MRN": "Midbrain",
    "SNr": "Midbrain", "VTA": "Midbrain", "PAG": "Midbrain",
    "APN": "Midbrain", "IC": "Midbrain", "RN": "Midbrain",
    "PPN": "Midbrain", "MB": "Midbrain",
    "BLA": "Amygdala", "BMA": "Amygdala", "MEA": "Amygdala",
    "LA": "Amygdala", "EP": "Amygdala", "EPd": "Amygdala",
    "CEA": "Amygdala",
    "LS": "Septum", "LSc": "Septum", "LSr": "Septum",
    "MS": "Septum", "TRS": "Septum",
    "PIR": "Olfactory", "AON": "Olfactory", "OLF": "Olfactory",
    "TT": "Olfactory",
    "root": "Other", "void": "Other",
}

GROUP_COLOR = {
    "Cortex":      "#D55E00",
    "HPF":         "#0072B2",
    "Striatum":    "#009E73",
    "Pallidum":    "#56B4E9",
    "Thalamus":    "#CC79A7",
    "Hypothalamus":"#999999",
    "Midbrain":    "#F0E442",
    "Amygdala":    "#882255",
    "Septum":      "#117733",
    "Olfactory":   "#E69F00",
    "Other":       "#BBBBBB",
}


def main():
    apply_style()

    stats_path = (
        PROJECT_ROOT / "outputs" / "eval_local"
        / "region_hierarchy_stats.json"
    )
    stats = json.load(open(stats_path))
    p1l = stats["per_region"]["1L SNN"]
    p2l = stats["per_region"].get("2L SNN", {})

    # Sort regions by 1L mean
    regions = sorted(p1l.keys(), key=lambda r: -p1l[r]["mean"])
    n = len(regions)

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 4.5))

    means_1l = [p1l[r]["mean"] for r in regions]
    ses_1l = [p1l[r]["se"] for r in regions]
    means_2l = [p2l.get(r, {}).get("mean", np.nan) for r in regions]
    colors = [
        GROUP_COLOR.get(REGION_GROUP.get(r, "Other"), "#BBBBBB")
        for r in regions
    ]

    y = np.arange(n)[::-1]
    ax.barh(
        y, means_1l, color=colors, edgecolor="none", height=0.75,
        alpha=0.85, zorder=2,
    )
    ax.errorbar(
        means_1l, y, xerr=ses_1l, fmt="none",
        ecolor="#444444", elinewidth=0.6, capsize=1.5, zorder=3,
    )
    # 2L SNN as small black ticks
    ax.scatter(
        means_2l, y, color="#222222", s=8, marker="|",
        linewidths=1.0, zorder=4, label="2L SNN",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r} ({p1l[r]['n']})" for r in regions], fontsize=6.5,
    )
    ax.set_xlabel(
        "Per-neuron Pearson $r$ (mean ± SE; bars = 1L SNN, ticks = 2L SNN)",
        fontsize=8.5,
    )
    ax.set_title(
        f"Brain-region predictability hierarchy — {n} regions, "
        f"27{{,}}144 neurons (Steinmetz 39 sessions)",
        fontsize=9, loc="left",
    )

    # Group color legend (compact)
    used_groups = list(dict.fromkeys(
        REGION_GROUP.get(r, "Other") for r in regions
    ))
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR.get(g, "#BBBBBB"))
        for g in used_groups
    ]
    ax.legend(
        handles, used_groups, loc="lower right",
        fontsize=7, ncol=2, frameon=False, title="Functional group",
        title_fontsize=7,
    )
    ax.set_xlim(0, max(means_1l) * 1.05)

    # K-W stat callout
    kw1 = stats["kruskal_wallis"]["1L SNN"]
    ax.text(
        0.02, -0.08,
        f"Kruskal–Wallis $H = {kw1['H']:.0f}$, $p < 10^{{-300}}$  "
        f"(across {kw1['n_regions']} regions, {kw1['n_neurons_total']:,} neurons)",
        transform=ax.transAxes, fontsize=8, color="#222222",
    )

    out_dir = (
        PROJECT_ROOT / "docs" / "neurips_neurocog" / "figures" / "mockups"
    )
    save_figure(fig, "region_v2_grouped_bars", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
