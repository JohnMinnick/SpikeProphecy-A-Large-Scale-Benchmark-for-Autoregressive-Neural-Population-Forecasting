"""Hero figure mockup A: minimalist forest plot with denoising/tax shading.

Single-panel design. X-axis = trial-vote response accuracy. Y-axis =
architectures (Transformer/Mamba/LRU/SNN) + raw-count anchor + chance.
A vertical dashed line at the raw-counts baseline; right of it shaded
green ("implicit denoising"), left shaded red ("decoding tax"). Two
callout arrows mark the headline effects.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from figures.style import apply_style, save_figure, TEXT_WIDTH, COLORS


def get_acc(d, key, sub):
    if key == "trial_level":
        return d.get("trial_level", {}).get(sub, np.nan)
    return d.get(key, {}).get(sub, np.nan)


def main():
    rd = PROJECT_ROOT / "outputs" / "eval_local"
    decoders = [
        ("linear_steinmetz", "Raw spike counts", "#444444"),
        ("transformer", "Transformer", COLORS["Transformer"]),
        ("mamba", "Mamba", COLORS["Mamba"]),
        ("lru", "LRU", COLORS["LRU"]),
        ("snn_standalone_v12b", "Spiking NN", COLORS["SNN"]),
    ]

    # Load per-session for error bars (use std across sessions)
    rows = []
    for tag, label, color in decoders:
        p = rd / f"behavioral_decode_{tag}.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        # response trial vote per session: count_correct/n trials in each session
        # the JSON has trial-level aggregated only, so use per_session resp_acc
        # as a proxy for variability
        sessions = d.get("per_session", [])
        if sessions:
            sess_resp = np.array([s.get("resp_acc", np.nan) for s in sessions])
            sess_resp = sess_resp[~np.isnan(sess_resp)]
        else:
            sess_resp = np.array([])
        agg = d.get("trial_level", {}).get("resp_3_majority", np.nan)
        rows.append((tag, label, color, agg, sess_resp))

    # Sort: anchors first, then ANNs by accuracy, then SNN
    raw_acc = next(r[3] for r in rows if r[0] == "linear_steinmetz")

    apply_style()
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.8))

    n = len(rows)
    y = np.arange(n)[::-1]

    # Background shading: green right of raw, red left of raw
    ax.add_patch(Rectangle(
        (raw_acc, -0.5), 1 - raw_acc, n,
        facecolor="#d8eed4", alpha=0.35, edgecolor="none", zorder=0,
    ))
    ax.add_patch(Rectangle(
        (0, -0.5), raw_acc, n,
        facecolor="#f4d4d4", alpha=0.30, edgecolor="none", zorder=0,
    ))

    # Vertical baseline line
    ax.axvline(
        raw_acc, color="#444444", linewidth=1.2, linestyle="--", zorder=1,
    )
    # Chance line
    ax.axvline(1 / 3, color="#bbbbbb", linewidth=0.7, linestyle=":", zorder=1)

    # Plot points
    for i, (tag, label, color, agg, sess) in enumerate(rows):
        # SE across sessions (bin-level resp_acc) as a proxy for trial-vote variability
        se = float(sess.std() / np.sqrt(len(sess))) if len(sess) > 1 else 0.0
        ax.errorbar(
            agg, y[i],
            xerr=se,
            fmt="o", color=color, markersize=7,
            ecolor=color, elinewidth=1.0, capsize=3,
            markeredgewidth=0,
            zorder=3,
        )
        # Label accuracy text
        ax.text(
            agg + 0.012, y[i],
            f"{agg*100:.1f}%",
            va="center", ha="left",
            fontsize=8, color="#222222", zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [r[1] for r in rows], fontsize=9,
    )
    ax.set_xlabel(
        "Trial-vote response decoding accuracy "
        "($p$ = predicted from each model; chance = 33.3%)",
        fontsize=8.5,
    )
    ax.set_xlim(0.30, 0.86)
    ax.set_ylim(-0.5, n - 0.5)

    # Header annotations
    ax.text(
        raw_acc + 0.02, n - 0.7,
        "implicit denoising",
        fontsize=8.5, color="#1a6d2a", style="italic",
        ha="left", va="bottom",
    )
    ax.text(
        raw_acc - 0.02, n - 0.7,
        "decoding tax",
        fontsize=8.5, color="#a02929", style="italic",
        ha="right", va="bottom",
    )
    ax.text(
        1/3 + 0.005, -0.4,
        "chance",
        fontsize=7, color="#999999",
        ha="left", va="top",
    )

    # Title
    ax.set_title(
        "Forecaster architecture determines behavioral decoding accuracy",
        fontsize=10, loc="left", pad=8,
    )

    # Strip top and right spines, plus left for forest-plot look
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#888888")

    out_dir = PROJECT_ROOT / "docs" / "neurips_neurocog" / "figures" / "mockups"
    save_figure(fig, "hero_A_forest", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
