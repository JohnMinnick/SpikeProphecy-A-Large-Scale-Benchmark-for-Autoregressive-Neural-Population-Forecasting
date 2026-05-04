"""Brain region predictability hierarchy figure for the Neuro&Cog paper.

Uses fresh per-neuron r values from outputs/eval_local/*_full.json (Steinmetz
sessions only — 39 sessions, ~27K neurons) and brain region annotations from
outputs/eval_analysis/brain_region_mapping.json.

Computes:
  - Per-region per-neuron r (mean ± SE) for one or more models
  - Kruskal-Wallis H test across regions
  - ANCOVA-corrected r (controlling for firing rate + Fano factor) [optional]

Outputs:
  - docs/neurips_neurocog/figures/figure_region_hierarchy.{png,pdf}
  - outputs/eval_local/region_hierarchy_stats.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from figures.style import apply_style, save_figure, TEXT_WIDTH, COLORS


# Allen CCF parent group mapping (rough cortical/subcortical/cerebellar)
# Used only for color-coding in plot.
REGION_GROUP = {
    # Isocortex
    "MOs": "Isocortex", "MOp": "Isocortex", "VISp": "Isocortex",
    "VISa": "Isocortex", "VISam": "Isocortex", "VISl": "Isocortex",
    "VISpm": "Isocortex", "VISrl": "Isocortex", "RSP": "Isocortex",
    "ACA": "Isocortex", "PL": "Isocortex", "ILA": "Isocortex",
    "ORB": "Isocortex", "AUD": "Isocortex", "SSp": "Isocortex",
    "SSs": "Isocortex", "TEa": "Isocortex", "AI": "Isocortex",
    "ECT": "Isocortex", "PERI": "Isocortex", "FRP": "Isocortex",
    "GU": "Isocortex", "DP": "Isocortex", "RSPv": "Isocortex",
    # Hippocampal formation
    "CA1": "HPF", "CA2": "HPF", "CA3": "HPF", "CA": "HPF",
    "DG": "HPF", "SUB": "HPF", "POST": "HPF", "PRE": "HPF",
    "PAR": "HPF", "ENT": "HPF", "ProS": "HPF",
    # Olfactory
    "PIR": "Olfactory", "AON": "Olfactory", "TT": "Olfactory",
    "MOB": "Olfactory", "AOB": "Olfactory",
    # Striatum / pallidum
    "CP": "Striatum", "ACB": "Striatum", "FS": "Striatum",
    "OT": "Striatum", "GPi": "Pallidum", "GPe": "Pallidum",
    "SI": "Pallidum",
    # Thalamus
    "LP": "Thalamus", "LD": "Thalamus", "MD": "Thalamus",
    "VPM": "Thalamus", "VPL": "Thalamus", "PO": "Thalamus",
    "VAL": "Thalamus", "RT": "Thalamus", "LGd": "Thalamus",
    "MG": "Thalamus", "AV": "Thalamus", "AM": "Thalamus",
    "AD": "Thalamus", "CL": "Thalamus", "PCN": "Thalamus",
    "PF": "Thalamus", "CM": "Thalamus", "MH": "Thalamus",
    "LH": "Thalamus",
    # Hypothalamus
    "ZI": "Hypothalamus", "PVH": "Hypothalamus",
    # Midbrain
    "SCm": "Midbrain", "SCs": "Midbrain", "MRN": "Midbrain",
    "SNr": "Midbrain", "SNc": "Midbrain", "VTA": "Midbrain",
    "PAG": "Midbrain", "APN": "Midbrain", "IC": "Midbrain",
    "RN": "Midbrain", "PPN": "Midbrain", "MB": "Midbrain",
    # Hindbrain
    "PRNc": "Hindbrain", "PRNr": "Hindbrain",
    # Amygdala
    "BLA": "Amygdala", "BMA": "Amygdala", "MEA": "Amygdala",
    "LA": "Amygdala", "COA": "Amygdala", "EP": "Amygdala",
    "EPd": "Amygdala", "CEA": "Amygdala",
    # Septum / etc
    "LS": "Septum", "LSc": "Septum", "LSr": "Septum",
    "MS": "Septum", "TRS": "Septum",
    # Other / root
    "root": "Other", "void": "Other",
}

GROUP_COLOR = {
    "Isocortex":   "#D55E00",  # vermillion
    "HPF":         "#0072B2",  # blue
    "Olfactory":   "#E69F00",  # orange
    "Striatum":    "#009E73",  # green
    "Pallidum":    "#56B4E9",  # sky blue
    "Thalamus":    "#CC79A7",  # purple
    "Hypothalamus":"#999999",
    "Midbrain":    "#F0E442",  # yellow
    "Hindbrain":   "#666666",
    "Amygdala":    "#882255",
    "Septum":      "#117733",
    "Other":       "#BBBBBB",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--eval-jsons",
        nargs="+",
        required=True,
        help="Per-session eval JSONs (one per model, with per_neuron_r per session).",
    )
    p.add_argument(
        "--model-names",
        nargs="+",
        required=True,
        help="Names matching eval-jsons.",
    )
    p.add_argument(
        "--region-mapping",
        type=str,
        default=str(
            PROJECT_ROOT / "outputs" / "eval_analysis"
            / "brain_region_mapping.json"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "docs" / "neurips_neurocog" / "figures"),
    )
    p.add_argument(
        "--name", type=str, default="figure_region_hierarchy"
    )
    p.add_argument(
        "--min-neurons",
        type=int,
        default=30,
        help="Min neurons per region to include in figure.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Show top-N regions by mean r in the dot plot.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    region_map_path = Path(args.region_mapping)
    region_map = json.load(open(region_map_path))
    sessions_map = region_map["sessions"]  # keyed by session index str

    # Build per-model: per-region list of per-neuron r values
    # model_data[model][region] = list of r values
    model_data = {name: defaultdict(list) for name in args.model_names}
    n_sessions_used = 0
    n_neurons_used = 0

    for model_name, eval_path in zip(args.model_names, args.eval_jsons):
        d = json.load(open(eval_path))
        ps = d["per_session"]
        for s in ps:
            sidx = s["session_idx"]
            if str(sidx) not in sessions_map:
                continue
            regions = sessions_map[str(sidx)].get("neuron_regions")
            if regions is None:
                continue
            r_list = s.get("per_neuron_r")
            if r_list is None:
                continue
            n_neur = min(len(regions), len(r_list))
            for i in range(n_neur):
                rg = regions[i]
                model_data[model_name][rg].append(float(r_list[i]))
            if model_name == args.model_names[0]:
                n_sessions_used += 1
                n_neurons_used += n_neur

    print(
        f"Sessions covered (model 0): {n_sessions_used}  "
        f"neurons covered: {n_neurons_used}"
    )

    # Filter regions by min_neurons (use first model's coverage as anchor)
    anchor = model_data[args.model_names[0]]
    region_n = {rg: len(rs) for rg, rs in anchor.items()}
    keep = [rg for rg, n in region_n.items() if n >= args.min_neurons]
    print(f"Regions after min_neurons={args.min_neurons} filter: {len(keep)}")

    # Per-region stats per model
    stats_per_model = {}
    for name, data in model_data.items():
        stats = {}
        for rg in keep:
            rs = np.array(data.get(rg, []))
            if len(rs) == 0:
                continue
            stats[rg] = {
                "n": int(len(rs)),
                "mean": float(np.mean(rs)),
                "median": float(np.median(rs)),
                "std": float(np.std(rs)),
                "se": float(np.std(rs) / np.sqrt(len(rs))),
                "p25": float(np.percentile(rs, 25)),
                "p75": float(np.percentile(rs, 75)),
            }
        stats_per_model[name] = stats

    # Kruskal-Wallis test per model: across the regions
    kw_results = {}
    for name, data in model_data.items():
        groups = [data[rg] for rg in keep if len(data[rg]) > 0]
        if len(groups) >= 2:
            H, p = kruskal(*groups)
            kw_results[name] = {
                "H": float(H),
                "p": float(p),
                "n_regions": len(groups),
                "n_neurons_total": int(sum(len(g) for g in groups)),
            }
            print(
                f"  {name}: K-W H={H:.1f} p={p:.3e} "
                f"(n_regions={len(groups)}, n_neurons={sum(len(g) for g in groups)})"
            )
        else:
            kw_results[name] = None

    # ----- Plot: dumbbell of top-N regions sorted by anchor model's r -----
    apply_style()
    anchor_stats = stats_per_model[args.model_names[0]]
    sorted_regions = sorted(
        anchor_stats.keys(), key=lambda r: -anchor_stats[r]["mean"]
    )[: args.top_n]

    n_models = len(args.model_names)
    height = max(3.0, 0.18 * len(sorted_regions) + 0.6)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, height))

    y = np.arange(len(sorted_regions))[::-1]  # invert so highest at top

    # Markers per model
    markers = ["D", "o", "s", "^", "v"][:n_models]
    palette = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#999999"][:n_models]
    if n_models == 1:
        palette = [COLORS["SNN"]]
        markers = ["o"]

    for mi, name in enumerate(args.model_names):
        st = stats_per_model[name]
        means = [st.get(r, {}).get("mean", np.nan) for r in sorted_regions]
        ses = [st.get(r, {}).get("se", 0.0) for r in sorted_regions]
        ax.errorbar(
            means, y,
            xerr=ses,
            fmt=markers[mi],
            color=palette[mi],
            ecolor=palette[mi],
            elinewidth=0.8,
            capsize=2.0,
            label=name,
            markersize=4.5,
            markeredgewidth=0,
            alpha=0.95,
        )

    # Region color strip on left
    for yi, rg in enumerate(sorted_regions):
        grp = REGION_GROUP.get(rg, "Other")
        c = GROUP_COLOR.get(grp, "#BBBBBB")
        ax.barh(
            y[yi], width=0.001, left=-0.01, height=0.85,
            color=c, edgecolor="none", zorder=0,
        )

    # Y axis labels: region + n
    ax.set_yticks(y)
    n_anchor = stats_per_model[args.model_names[0]]
    ylabels = [
        f"{r}  (n={n_anchor[r]['n']})" for r in sorted_regions
    ]
    ax.set_yticklabels(ylabels, fontsize=7.5)

    ax.set_xlabel(
        f"Per-neuron Pearson $r$ (mean $\\pm$ SE)", fontsize=9
    )
    ax.axvline(0, color="#999999", linewidth=0.6, linestyle=":")
    ax.set_xlim(left=-0.01)

    # Annotate K-W
    kw_lines = []
    for name in args.model_names:
        kw = kw_results.get(name)
        if kw:
            kw_lines.append(
                f"{name}: $H$={kw['H']:.0f}, $p$<{max(kw['p'], 1e-300):.0e}"
            )
    if kw_lines:
        ax.text(
            0.50, -0.15, "  |  ".join(kw_lines),
            transform=ax.transAxes, fontsize=7.5,
            ha="center", va="top",
            color="#222222",
        )

    if n_models > 1:
        ax.legend(
            loc="upper right", fontsize=7.5, frameon=False,
            bbox_to_anchor=(1.0, 1.0),
        )

    ax.set_title(
        f"Brain region predictability (top {len(sorted_regions)} of "
        f"{len(keep)} regions; Steinmetz 39-session benchmark)",
        fontsize=9, loc="left",
    )

    out_dir = Path(args.out_dir)
    save_figure(fig, args.name, out_dir=out_dir)
    plt.close(fig)

    # Save stats JSON
    stats_out = {
        "model_names": args.model_names,
        "n_sessions_steinmetz": n_sessions_used,
        "n_neurons_total": n_neurons_used,
        "min_neurons_per_region": args.min_neurons,
        "n_regions_after_filter": len(keep),
        "kruskal_wallis": kw_results,
        "per_region": stats_per_model,
    }
    out_json = (
        PROJECT_ROOT / "outputs" / "eval_local" / "region_hierarchy_stats.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"  Stats: {out_json}")


if __name__ == "__main__":
    main()
