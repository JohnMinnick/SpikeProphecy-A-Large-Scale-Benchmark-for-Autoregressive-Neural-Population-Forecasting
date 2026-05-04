"""F5 synthetic-population validation variants.

  v1_existing_panel: copy the existing four-panel figure
  v2_ks_summary_bar: one-panel summary bar with KS D per statistic
  v3_dist_overlay: CDF overlay for firing rate + Fano
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


def _load_stats():
    p = PROJECT_ROOT / "outputs" / "eval_local" / "synthetic_validation_stats.json"
    return json.load(open(p))


def v1_existing_panel(out_dir):
    import shutil
    src = PROJECT_ROOT / "docs" / "neurips_neurocog" / "figures" / "figure_synthetic_validation.png"
    if src.exists():
        shutil.copy(src, out_dir / "v1_existing_panel.png")
        pdf_src = src.with_suffix(".pdf")
        if pdf_src.exists():
            shutil.copy(pdf_src, out_dir / "v1_existing_panel.pdf")
        print("  v1: copied existing 4-panel figure")


def v2_ks_summary_bar(stats, out_dir):
    """KS D per statistic, Mamba vs SNN."""
    apply_style()
    ks = stats["ks_tests"]
    pair = stats["pairwise_corr_recovery"]

    statistics = [
        ("firing_rate", "Firing rate"),
        ("fano", "Fano factor"),
        ("pop_rate", "Population rate"),
    ]
    mamba_ks = [ks[k[0]]["mamba_vs_gt"]["D"] for k in statistics]
    snn_ks = [ks[k[0]]["snn_vs_gt"]["D"] for k in statistics]

    fig, axes = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH * 0.9, 2.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.3, 1.0]},
    )

    # Left: KS bars
    ax = axes[0]
    x = np.arange(len(statistics))
    width = 0.38
    ax.bar(
        x - width / 2, mamba_ks, width, color=COLORS["Mamba"], alpha=0.9,
        label="Mamba", edgecolor="white", linewidth=0.4,
    )
    ax.bar(
        x + width / 2, snn_ks, width, color=COLORS["SNN"], alpha=0.9,
        label="Spiking NN", edgecolor="white", linewidth=0.4,
    )
    ax.axhline(0.1, color="#888", ls="--", lw=0.7, alpha=0.7)
    ax.text(
        2.3, 0.105, "threshold = 0.1", fontsize=7, color="#888",
        va="bottom",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([s[1] for s in statistics], fontsize=8.5)
    ax.set_ylabel("KS $D$ (smaller = better)", fontsize=9)
    ax.set_ylim(0, max(max(mamba_ks), max(snn_ks)) + 0.07)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title(
        "Marginal distributional fidelity",
        fontsize=9.5, loc="left",
    )

    # Right: pairwise corr recovery
    ax = axes[1]
    ax.bar(
        [0, 1],
        [pair["mamba_vs_gt"]["mean"], pair["snn_vs_gt"]["mean"]],
        color=[COLORS["Mamba"], COLORS["SNN"]], alpha=0.9,
        edgecolor="white", linewidth=0.5,
    )
    ax.errorbar(
        [0, 1],
        [pair["mamba_vs_gt"]["mean"], pair["snn_vs_gt"]["mean"]],
        yerr=[pair["mamba_vs_gt"]["std"], pair["snn_vs_gt"]["std"]],
        fmt="none", ecolor="#333", capsize=3, lw=0.8,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Mamba", "SNN"], fontsize=9)
    ax.set_ylabel("Pearson $r$ (off-diag entries)", fontsize=9)
    ax.set_title(
        "Pairwise correlation\nmatrix recovery",
        fontsize=9.5, loc="left",
    )
    ax.set_ylim(0, 0.25)

    save_figure(fig, "v2_ks_summary_bar", out_dir=out_dir)
    plt.close(fig)


def v3_dist_overlay(out_dir):
    """Overlay CDFs of firing rate + Fano using cached inference arrays."""
    apply_style()
    inf_dir = PROJECT_ROOT / "outputs" / "full_inference"
    if not inf_dir.exists():
        return

    # Aggregate per-neuron rates across sessions (Steinmetz only)
    gt_rates = []
    mamba_rates = []
    snn_rates = []
    rng = np.random.default_rng(42)
    for sess_file in sorted(inf_dir.glob("session_*.npz"))[:20]:
        d = np.load(sess_file)
        m_act = int(d["m_actual"])
        if m_act < 50:
            continue
        gt = d["gt"][:m_act]
        mamba = d["mamba_rates"][:m_act]
        snn = d["snn_rates"][:m_act]
        gt_rates.append(gt.mean(axis=1))  # per-neuron mean rate
        mamba_rates.append(
            rng.poisson(np.clip(mamba, 0, 50)).astype(np.float32).mean(axis=1)
        )
        snn_rates.append(
            rng.poisson(np.clip(snn, 0, 50)).astype(np.float32).mean(axis=1)
        )
    gt_rates = np.concatenate(gt_rates)
    mamba_rates = np.concatenate(mamba_rates)
    snn_rates = np.concatenate(snn_rates)

    fig, ax = plt.subplots(
        figsize=(TEXT_WIDTH * 0.7, 2.5),
        constrained_layout=True,
    )

    def cdf(x, color, label, lw=1.2):
        x = np.sort(x)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=color, lw=lw, label=label)

    cdf(gt_rates, "#222", "Ground truth", lw=1.6)
    cdf(mamba_rates, COLORS["Mamba"], "Mamba (Poisson sampled)")
    cdf(snn_rates, COLORS["SNN"], "SNN (Poisson sampled)")
    ax.set_xlabel("Per-neuron mean firing rate (spikes/bin)", fontsize=9)
    ax.set_ylabel("Cumulative density", fontsize=9)
    ax.set_title(
        "Per-neuron firing-rate CDF (Steinmetz 20 sessions)",
        fontsize=9.5, loc="left",
    )
    ax.set_xlim(0, 2)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    save_figure(fig, "v3_dist_overlay", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = (
        PROJECT_ROOT / "docs" / "neurips_neurocog"
        / "figure_candidates" / "F5_synthetic"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = _load_stats()
    v1_existing_panel(out_dir)
    v2_ks_summary_bar(stats, out_dir)
    v3_dist_overlay(out_dir)
    print(f"Wrote {len(list(out_dir.glob('*.png')))} PNGs to {out_dir}")
