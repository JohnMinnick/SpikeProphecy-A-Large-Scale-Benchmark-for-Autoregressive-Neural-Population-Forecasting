"""Cross-architecture behavioral decodability figure.

Reads per-architecture decode JSONs from outputs/eval_local/ and produces
a forest-plot or bar chart with 95% CIs comparing how well behavioral
variables can be decoded from each model's rate predictions vs. the
raw-count linear baseline.
"""

import argparse
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


# Family-coded readouts: same color within family (baselines / Mamba / SNN),
# same marker within family.  Three baselines run light-to-dark grey;
# Mamba seeds share the Mamba vermillion (style.py COLORS['Mamba']) at
# different fill alphas; SNN seeds share the SNN green (style.py
# COLORS['SNN']) at different fill alphas.  Markers: circle = baseline,
# diamond = Mamba (matches MARKERS['Mamba']), X = SNN (matches MARKERS['SNN']).
DEFAULT_READOUTS = [
    # tag, label, color, marker, fill_alpha
    ("linear_steinmetz",     "Linear / 1-bin raw counts", "#999999",          "o", 1.0),
    ("raw_h10_sum",          "Linear / H=10 raw (sum)",   "#666666",          "o", 1.0),
    ("raw_h10_flat",         "Ridge / H=10 raw (flat)",   "#333333",          "o", 1.0),
    ("mamba",                "Mamba (s42)",               COLORS["Mamba"],    "D", 1.0),
    ("mamba_s1",             "Mamba (s1)",                COLORS["Mamba"],    "D", 0.55),
    ("mamba_s2",             "Mamba (s2)",                COLORS["Mamba"],    "D", 0.30),
    ("transformer",          "Transformer (s42)",         COLORS["Transformer"], "^", 1.0),
    ("transformer_s1",       "Transformer (s1)",          COLORS["Transformer"], "^", 0.55),
    ("transformer_s2",       "Transformer (s2)",          COLORS["Transformer"], "^", 0.30),
    ("lru",                  "LRU (s42)",                 COLORS["LRU"],      "s", 1.0),
    ("lru_s1",               "LRU (s1)",                  COLORS["LRU"],      "s", 0.55),
    ("lru_s2",               "LRU (s2)",                  COLORS["LRU"],      "s", 0.30),
    ("ndt2_style_s42",       "NDT2-style (bidir+mask)",   "#882255",          "v", 1.0),
    ("snn_standalone_v12b",  "SNN preds (s42)",           COLORS["SNN"],      "X", 1.0),
    ("snn_s1",               "SNN preds (s1)",            COLORS["SNN"],      "X", 0.55),
    ("snn_s2",               "SNN preds (s2)",            COLORS["SNN"],      "X", 0.30),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "eval_local"),
    )
    p.add_argument(
        "--readouts",
        nargs="+",
        default=None,
        help="List of tags to include (default: all that exist).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(
            PROJECT_ROOT / "docs" / "neurips_neurocog" / "figures"
        ),
    )
    p.add_argument(
        "--name",
        type=str,
        default="figure_cross_arch_decoding",
    )
    return p.parse_args()


def load_decode(tag: str, results_dir: Path) -> dict | None:
    p = results_dir / f"behavioral_decode_{tag}.json"
    if not p.exists():
        return None
    return json.load(open(p))


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    # Pick readouts
    readouts = []
    for tag, label, color, marker, alpha in DEFAULT_READOUTS:
        if args.readouts and tag not in args.readouts:
            continue
        d = load_decode(tag, results_dir)
        if d is None:
            print(f"  skip missing: {tag}")
            continue
        readouts.append((tag, label, color, marker, alpha, d))

    if not readouts:
        print("No readouts found")
        return

    # Three metrics to plot
    metrics = [
        ("response_bin", "Response 3-class\n(bin)", 1 / 3),
        ("stimulus_bin_16class", "Stimulus 16-class\n(bin)", 1 / 16),
        ("stimulus_bin_side3", "Stim side 3-class\n(bin)", 1 / 3),
    ]
    trial_keys = [
        ("resp_3_majority", "Response 3-class\n(trial vote)", 1 / 3),
        ("stim_16_majority", "Stimulus 16-class\n(trial vote)", 1 / 16),
        ("side_3_majority", "Stim side 3-class\n(trial vote)", 1 / 3),
    ]

    apply_style()
    fig, axes = plt.subplots(
        2, 3,
        figsize=(TEXT_WIDTH, 3.6),
        constrained_layout=True,
        sharey=False,
    )

    n_readouts = len(readouts)
    y = np.arange(n_readouts)[::-1]  # invert so first is at top

    # Pre-compute family-band y-ranges for shaded background grouping.
    # Identify contiguous groups by family from the readout tags.
    def _family(tag: str) -> str:
        if tag.startswith(("linear", "raw_h10")):
            return "baseline"
        if tag.startswith("mamba"):
            return "mamba"
        if tag.startswith("transformer"):
            return "transformer"
        if tag.startswith("lru"):
            return "lru"
        if tag.startswith("ndt2"):
            return "ndt2"
        if tag.startswith("snn"):
            return "snn"
        return "other"

    families = [_family(r[0]) for r in readouts]
    # y-coordinates in display order (top-to-bottom)
    y_display = list(y)
    # Find runs of same family for background banding
    bands = []  # list of (family, y_top, y_bot)
    i = 0
    while i < n_readouts:
        j = i
        while j + 1 < n_readouts and families[j + 1] == families[i]:
            j += 1
            if j - i >= 5:
                break
        # y values are inverted (first row at top); top = max(y[i:j+1])
        bands.append((families[i], max(y_display[i:j + 1]), min(y_display[i:j + 1])))
        i = j + 1

    BAND_COLORS = {
        "baseline":    "#F4F4F4",
        "mamba":       "#FFEBE0",
        "transformer": "#F5E6F0",
        "lru":         "#E6F0F7",
        "ndt2":        "#F0E0EC",
        "snn":         "#E2F4EC",
    }

    def _draw_bands(ax):
        for fam, ytop, ybot in bands:
            ax.axhspan(
                ybot - 0.45, ytop + 0.45,
                color=BAND_COLORS.get(fam, "#FAFAFA"),
                zorder=0, linewidth=0,
            )

    # --- Row 1: bin-level accuracy ---
    for col, (key, title, chance) in enumerate(metrics):
        ax = axes[0, col]
        _draw_bands(ax)
        for i, (tag, label, color, marker, alpha, d) in enumerate(readouts):
            m = d.get(key)
            if m is None:
                continue
            mean = m["acc"]
            lo = m["ci95_lo"]
            hi = m["ci95_hi"]
            ax.errorbar(
                mean, y[i],
                xerr=[[mean - lo], [hi - mean]],
                fmt=marker,
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                ecolor=color,
                alpha=alpha,
                markersize=5,
                capsize=2.0,
                linewidth=0.8,
                markeredgewidth=0.4,
                zorder=3,
            )
        ax.axvline(
            chance, color="#aaaaaa", linestyle=":", linewidth=0.6,
            zorder=1,
        )
        ax.set_title(title, fontsize=8.5)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [r[1] for r in readouts] if col == 0 else [],
            fontsize=7.0,
        )
        ax.tick_params(labelsize=7)
        # Tighten x-axis to where the data lives
        all_his = [d.get(key, {}).get("ci95_hi", np.nan) for _, _, _, _, _, d in readouts]
        x_max = max(0.70, np.nanmax(all_his) + 0.04)
        ax.set_xlim(0, x_max)
        # Hide spine clutter on bands
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # --- Row 2: trial-level majority vote ---
    for col, (key, title, chance) in enumerate(trial_keys):
        ax = axes[1, col]
        _draw_bands(ax)
        all_means = []
        for i, (tag, label, color, marker, alpha, d) in enumerate(readouts):
            tl = d.get("trial_level", {})
            mean = tl.get(key, np.nan)
            if mean is None or (isinstance(mean, float) and np.isnan(mean)):
                continue
            all_means.append(mean)
            ax.scatter(
                mean, y[i],
                color=color, marker=marker,
                alpha=alpha, s=36,
                edgecolors=color, linewidths=0.4,
                zorder=3,
            )
        ax.axvline(
            chance, color="#aaaaaa", linestyle=":", linewidth=0.6,
            zorder=1,
        )
        ax.set_title(title, fontsize=8.5)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [r[1] for r in readouts] if col == 0 else [],
            fontsize=7.0,
        )
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Accuracy", fontsize=8)
        ax.set_xlim(0, max(0.90, max(all_means) + 0.04 if all_means else 0.5))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle(
        "Behavioral decodability from Mamba's predicted rates vs. "
        "matched-context raw counts "
        "(Steinmetz 39 sessions, 1,994 held-out trials)",
        fontsize=9, y=1.02,
    )

    out_dir = Path(args.out_dir)
    save_figure(fig, args.name, out_dir=out_dir)
    plt.close(fig)

    # Also print a summary table
    print()
    print("=" * 90)
    print(
        f"{'Readout':>22s}  "
        f"{'Resp bin':>10s}  {'Stim16 bin':>12s}  {'Side3 bin':>11s}  "
        f"{'Resp vote':>10s}  {'Stim16 vote':>12s}  {'Side3 vote':>11s}"
    )
    print("=" * 90)
    for tag, label, color, marker, alpha, d in readouts:
        rb = d["response_bin"]["acc"]
        sb = d["stimulus_bin_16class"]["acc"]
        side = d["stimulus_bin_side3"]["acc"]
        tl = d.get("trial_level", {})
        rv = tl.get("resp_3_majority", np.nan)
        sv = tl.get("stim_16_majority", np.nan)
        sdv = tl.get("side_3_majority", np.nan)
        print(
            f"{label:>22s}  "
            f"{rb:>10.3f}  {sb:>12.3f}  {side:>11.3f}  "
            f"{rv:>10.3f}  {sv:>12.3f}  {sdv:>11.3f}"
        )


if __name__ == "__main__":
    main()
