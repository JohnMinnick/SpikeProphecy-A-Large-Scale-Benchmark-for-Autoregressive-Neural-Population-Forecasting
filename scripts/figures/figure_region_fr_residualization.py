"""Visual: region ranking is preserved after firing-rate residualization.

Reads outputs/eval_local/region_hierarchy_residualized.json and produces
a scatter / paired-bar comparison of raw vs residualized region rankings.
For the rebuttal: shows visually that the region hierarchy is not
reducible to mean firing-rate differences across regions.
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
        / "region_hierarchy_residualized.json"
    )
    if not src.exists():
        print("missing", src)
        return
    data = json.load(open(src))

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.8),
                             constrained_layout=True)

    for ax, (model_name, mdata) in zip(axes, data["models"].items()):
        raw_means = mdata["per_region_mean_raw_r"]
        res_means = mdata["per_region_mean_residualized_r"]
        common = sorted(set(raw_means) & set(res_means))
        x = np.array([raw_means[r] for r in common])
        y = np.array([res_means[r] for r in common])

        ax.scatter(x, y, s=10, color="#0072B2", alpha=0.7, edgecolors="none")
        # Diagonal-ish reference: linear fit to show alignment
        if len(x) > 2:
            slope, intercept = np.polyfit(x, y, 1)
            xmin, xmax = float(np.min(x)), float(np.max(x))
            xs = np.linspace(xmin, xmax, 100)
            ax.plot(xs, slope * xs + intercept, "--", color="#666", lw=0.7)

        rho = mdata["spearman_rho_raw_vs_residualized"]
        top10 = mdata["top10_overlap"]
        ax.set_title(
            f"{model_name}: " + r"Spearman $\rho=" + f"{rho:.3f}$, "
            + f"top-10 overlap = {top10}/10",
            fontsize=8.5,
        )
        ax.set_xlabel("Per-region mean per-neuron $r$ (raw)", fontsize=8)
        ax.set_ylabel(
            "Per-region mean per-neuron $r$\n(residualized vs $\\log_{10}$ FR)",
            fontsize=8,
        )
        ax.tick_params(labelsize=7)

    fig.suptitle(
        "Region predictability hierarchy is preserved after firing-rate "
        "residualization (NeurIPS reviewer 2 ask)",
        fontsize=9, y=1.04,
    )

    out_dir = PROJECT_ROOT / "docs" / "neurips_neurocog" / "figures"
    save_figure(fig, "figure_region_fr_residualization", out_dir=out_dir)
    plt.close(fig)
    print("Saved: figure_region_fr_residualization.{png,pdf}")


if __name__ == "__main__":
    main()
