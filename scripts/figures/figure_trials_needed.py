"""Trials-needed-for-readout figure (NeurIPS reviewer 2 Q7).

Shows accuracy vs number of training trials per session for the
per-session linear readout fit on Mamba's predicted rates. Three
behavioral targets, mean +/- SEM across 39 sessions x 5 random
training-subset draws.
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


def main():
    src = (
        PROJECT_ROOT / "outputs" / "eval_local"
        / "trials_needed_for_readout.json"
    )
    if not src.exists():
        print("missing", src)
        return
    d = json.load(open(src))

    # Sort by train_frac
    items = sorted(d.items(), key=lambda kv: float(kv[0]))
    train_trials = np.array([v["n_train_trials_median"] for _, v in items])

    targets = [
        ("resp", "Response (3-class)", "#0072B2", 1 / 3),
        ("stim", "Stim 16-class", "#D55E00", 1 / 16),
        ("side", "Stim side (3-class)", "#009E73", 1 / 3),
    ]

    apply_style()
    fig, ax = plt.subplots(
        figsize=(TEXT_WIDTH * 0.65, 2.4), constrained_layout=True,
    )
    for tag, label, color, chance in targets:
        means = np.array([v[f"{tag}_trial_acc_mean"] for _, v in items])
        sems = np.array([v[f"{tag}_trial_acc_sem"] for _, v in items])
        ax.errorbar(
            train_trials, means, yerr=sems,
            fmt="o-", color=color, markersize=4, capsize=2.5,
            linewidth=1.0, markeredgewidth=0, label=label,
        )
        ax.axhline(chance, color=color, linestyle=":", linewidth=0.5,
                   alpha=0.5)

    ax.set_xlabel(
        "Training trials per session (median across 39 Steinmetz sessions)",
        fontsize=8,
    )
    ax.set_ylabel("Trial-vote accuracy", fontsize=8)
    ax.set_xscale("log")
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title(
        "Per-session linear readout convergence on Mamba rate predictions",
        fontsize=8.5,
    )

    out_dir = PROJECT_ROOT / "docs" / "neurips_neurocog" / "figures"
    save_figure(fig, "figure_trials_needed", out_dir=out_dir)
    plt.close(fig)
    print("Saved: figure_trials_needed.{png,pdf}")


if __name__ == "__main__":
    main()
