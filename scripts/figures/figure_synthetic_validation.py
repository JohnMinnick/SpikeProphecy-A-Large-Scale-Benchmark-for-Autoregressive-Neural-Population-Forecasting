"""Synthetic population validation: defends low per-neuron r via population stats.

Uses cached gt/mamba/snn rate arrays in outputs/full_inference/. For each
model, Poisson-samples rates to produce synthetic spike counts, then
compares distributional statistics to ground truth:

  (a) Per-neuron firing rate distribution        (CDF + KS test)
  (b) Per-neuron Fano factor distribution        (CDF + KS test)
  (c) Pairwise correlation matrix similarity      (corr of off-diag entries)
  (d) Population rate (sum across neurons)       (CDF + KS test)

Outputs:
  - docs/neurips_neurocog/figures/figure_synthetic_validation.{png,pdf}
  - outputs/eval_local/synthetic_validation_stats.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from figures.style import apply_style, save_figure, TEXT_WIDTH


def fano_factor(counts: np.ndarray) -> np.ndarray:
    """Per-neuron Fano factor = var(count) / mean(count). counts: (T, M)."""
    m = counts.mean(axis=0)
    v = counts.var(axis=0)
    f = np.where(m > 0, v / np.maximum(m, 1e-8), 0.0)
    return f


def pairwise_corr_matrix(
    counts: np.ndarray, max_neurons: int = 200
) -> np.ndarray:
    """Pearson correlation matrix between neurons (subsampled if needed)."""
    T, M = counts.shape
    if M > max_neurons:
        # Subsample by activity rank to keep top neurons
        rank = np.argsort(-counts.mean(axis=0))
        sel = np.sort(rank[:max_neurons])
        counts = counts[:, sel]
    # Pearson corr (numpy)
    c = counts.T - counts.T.mean(axis=1, keepdims=True)
    cov = c @ c.T / max(T - 1, 1)
    sd = np.sqrt(np.diag(cov))
    sd_safe = np.where(sd > 0, sd, 1.0)
    corr = cov / np.outer(sd_safe, sd_safe)
    return corr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inference-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "full_inference"),
    )
    p.add_argument(
        "--max-sessions",
        type=int,
        default=None,
    )
    p.add_argument(
        "--max-neurons-corr",
        type=int,
        default=200,
        help="Subsample to top-N neurons by rate for corr matrix.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "docs" / "neurips_neurocog" / "figures"),
    )
    p.add_argument(
        "--name", type=str, default="figure_synthetic_validation"
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    inf_dir = Path(args.inference_dir)
    files = sorted(inf_dir.glob("session_*.npz"))
    if args.max_sessions is not None:
        files = files[: args.max_sessions]
    print(f"Sessions to analyze: {len(files)}")

    # ---- Aggregate per-neuron stats across sessions ----
    gt_rates_list = []
    mamba_rates_list = []
    snn_rates_list = []
    gt_fano_list = []
    mamba_fano_list = []
    snn_fano_list = []
    pop_rates = {"gt": [], "mamba": [], "snn": []}
    corr_corrs = {"mamba_vs_gt": [], "snn_vs_gt": []}

    t0 = time.time()
    for f in files:
        d = np.load(f)
        m_act = int(d["m_actual"])
        if m_act < 30:
            continue
        gt = d["gt"][:, :m_act].astype(np.float32)
        mamba = d["mamba_rates"][:, :m_act].astype(np.float32)
        snn = d["snn_rates"][:, :m_act].astype(np.float32)

        # Poisson-sample model rates for fair count distribution comparison
        mamba_samp = rng.poisson(np.clip(mamba, 0, 50)).astype(np.float32)
        snn_samp = rng.poisson(np.clip(snn, 0, 50)).astype(np.float32)

        # (a) Per-neuron mean rate (in spikes/bin)
        gt_rates_list.append(gt.mean(axis=0))
        mamba_rates_list.append(mamba_samp.mean(axis=0))
        snn_rates_list.append(snn_samp.mean(axis=0))

        # (b) Per-neuron Fano factor
        gt_fano_list.append(fano_factor(gt))
        mamba_fano_list.append(fano_factor(mamba_samp))
        snn_fano_list.append(fano_factor(snn_samp))

        # (c) Pairwise corr matrix similarity (off-diagonal Pearson)
        gt_corr = pairwise_corr_matrix(gt, args.max_neurons_corr)
        mamba_corr = pairwise_corr_matrix(mamba_samp, args.max_neurons_corr)
        snn_corr = pairwise_corr_matrix(snn_samp, args.max_neurons_corr)
        # Take upper triangle (excluding diagonal)
        n = gt_corr.shape[0]
        iu = np.triu_indices(n, k=1)
        gt_off = gt_corr[iu]
        mamba_off = mamba_corr[iu]
        snn_off = snn_corr[iu]
        # Pearson r of off-diag entries (does the model recover the
        # pairwise-correlation structure?)
        if gt_off.std() > 0 and mamba_off.std() > 0:
            r_m = np.corrcoef(gt_off, mamba_off)[0, 1]
            corr_corrs["mamba_vs_gt"].append(float(r_m))
        if gt_off.std() > 0 and snn_off.std() > 0:
            r_s = np.corrcoef(gt_off, snn_off)[0, 1]
            corr_corrs["snn_vs_gt"].append(float(r_s))

        # (d) Population rate (sum across neurons over time)
        pop_rates["gt"].append(gt.sum(axis=1))
        pop_rates["mamba"].append(mamba_samp.sum(axis=1))
        pop_rates["snn"].append(snn_samp.sum(axis=1))

        elapsed = time.time() - t0
        print(
            f"  {f.name}: M={m_act}  T={gt.shape[0]}  | "
            f"corr_mamba_vs_gt={corr_corrs['mamba_vs_gt'][-1]:.3f}  "
            f"corr_snn_vs_gt={corr_corrs['snn_vs_gt'][-1]:.3f}  "
            f"({elapsed:.0f}s)"
        )

    # Concatenate across sessions
    gt_rates = np.concatenate(gt_rates_list)
    mamba_rates = np.concatenate(mamba_rates_list)
    snn_rates = np.concatenate(snn_rates_list)
    gt_fano = np.concatenate(gt_fano_list)
    mamba_fano = np.concatenate(mamba_fano_list)
    snn_fano = np.concatenate(snn_fano_list)
    pop_gt = np.concatenate(pop_rates["gt"])
    pop_mamba = np.concatenate(pop_rates["mamba"])
    pop_snn = np.concatenate(pop_rates["snn"])

    # ---- Compute KS statistics ----
    ks_results = {}
    for label, gt_a, mamba_a, snn_a in [
        ("firing_rate", gt_rates, mamba_rates, snn_rates),
        ("fano", gt_fano, mamba_fano, snn_fano),
        ("pop_rate", pop_gt, pop_mamba, pop_snn),
    ]:
        ks_m = ks_2samp(gt_a, mamba_a)
        ks_s = ks_2samp(gt_a, snn_a)
        ks_results[label] = {
            "mamba_vs_gt": {"D": float(ks_m.statistic), "p": float(ks_m.pvalue)},
            "snn_vs_gt": {"D": float(ks_s.statistic), "p": float(ks_s.pvalue)},
        }

    # ---- Plot 4-panel figure ----
    apply_style()
    fig, axes = plt.subplots(
        1, 4, figsize=(TEXT_WIDTH * 1.05, 2.2),
        constrained_layout=True,
    )

    # Color palette (Wong)
    GT_C = "#444444"
    MAMBA_C = "#D55E00"
    SNN_C = "#009E73"

    def cdf_panel(ax, gt_arr, mamba_arr, snn_arr, label, log=False):
        """Plot empirical CDF for three series."""
        # Cap extreme tail for log axis
        if log:
            gt_arr = gt_arr[gt_arr > 1e-6]
            mamba_arr = mamba_arr[mamba_arr > 1e-6]
            snn_arr = snn_arr[snn_arr > 1e-6]
        for a, c, lbl in [
            (gt_arr, GT_C, "GT"),
            (mamba_arr, MAMBA_C, "Mamba"),
            (snn_arr, SNN_C, "1L SNN"),
        ]:
            xs = np.sort(a)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(xs, ys, color=c, linewidth=1.2, label=lbl)
        ax.set_xlabel(label, fontsize=8.5)
        if log:
            ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.tick_params(labelsize=7.5)

    # (a) Firing rate CDF
    cdf_panel(
        axes[0], gt_rates, mamba_rates, snn_rates,
        "Per-neuron mean count", log=False,
    )
    axes[0].set_ylabel("CDF", fontsize=8.5)
    axes[0].set_title("(a) Firing rate", fontsize=9, loc="left")
    ksm = ks_results["firing_rate"]["mamba_vs_gt"]["D"]
    kss = ks_results["firing_rate"]["snn_vs_gt"]["D"]
    axes[0].text(
        0.55, 0.20,
        f"KS:\n Mamba = {ksm:.3f}\n SNN   = {kss:.3f}",
        transform=axes[0].transAxes, fontsize=7,
        family="monospace",
    )

    # (b) Fano factor CDF (cap to [0, 5] for visibility)
    fmax = 5.0
    cdf_panel(
        axes[1],
        np.clip(gt_fano, 0, fmax),
        np.clip(mamba_fano, 0, fmax),
        np.clip(snn_fano, 0, fmax),
        "Per-neuron Fano factor",
    )
    axes[1].set_title("(b) Fano factor", fontsize=9, loc="left")
    ksm = ks_results["fano"]["mamba_vs_gt"]["D"]
    kss = ks_results["fano"]["snn_vs_gt"]["D"]
    axes[1].text(
        0.55, 0.20,
        f"KS:\n Mamba = {ksm:.3f}\n SNN   = {kss:.3f}",
        transform=axes[1].transAxes, fontsize=7,
        family="monospace",
    )

    # (c) Population rate CDF
    cdf_panel(
        axes[2], pop_gt, pop_mamba, pop_snn,
        "Population count / bin",
    )
    axes[2].set_title("(c) Population rate", fontsize=9, loc="left")
    ksm = ks_results["pop_rate"]["mamba_vs_gt"]["D"]
    kss = ks_results["pop_rate"]["snn_vs_gt"]["D"]
    axes[2].text(
        0.55, 0.20,
        f"KS:\n Mamba = {ksm:.3f}\n SNN   = {kss:.3f}",
        transform=axes[2].transAxes, fontsize=7,
        family="monospace",
    )

    # (d) Pairwise corr matrix similarity (per-session r distribution)
    mamba_pcs = np.array(corr_corrs["mamba_vs_gt"])
    snn_pcs = np.array(corr_corrs["snn_vs_gt"])
    pos = [0, 1]
    bp = axes[3].boxplot(
        [mamba_pcs, snn_pcs],
        positions=pos,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=4),
        medianprops=dict(color="white", linewidth=1.4),
        whiskerprops=dict(color="#333333", linewidth=0.8),
        capprops=dict(color="#333333", linewidth=0.8),
        flierprops=dict(marker=".", markersize=2, alpha=0.5),
    )
    for patch, c in zip(bp["boxes"], [MAMBA_C, SNN_C]):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.6)
    axes[3].set_xticks(pos)
    axes[3].set_xticklabels(["Mamba", "1L SNN"], fontsize=8)
    axes[3].set_ylabel(
        "Per-session $r$\n(off-diag pairwise corr)", fontsize=8.5,
    )
    axes[3].set_title("(d) Pairwise correlation", fontsize=9, loc="left")
    axes[3].set_ylim(0, 1.0)
    axes[3].axhline(0, color="#999999", linewidth=0.4, linestyle=":")

    # Legend on first panel, placed above KS annotation
    axes[0].legend(
        loc="center right", fontsize=7, frameon=False,
        bbox_to_anchor=(1.0, 0.6),
    )

    out_dir = Path(args.out_dir)
    save_figure(fig, args.name, out_dir=out_dir)
    plt.close(fig)

    # ---- Save stats ----
    stats_out = {
        "n_sessions": len(files),
        "n_neurons_total": int(len(gt_rates)),
        "ks_tests": ks_results,
        "pairwise_corr_recovery": {
            "mamba_vs_gt": {
                "n_sessions": int(len(mamba_pcs)),
                "mean": float(np.mean(mamba_pcs)),
                "median": float(np.median(mamba_pcs)),
                "std": float(np.std(mamba_pcs)),
            },
            "snn_vs_gt": {
                "n_sessions": int(len(snn_pcs)),
                "mean": float(np.mean(snn_pcs)),
                "median": float(np.median(snn_pcs)),
                "std": float(np.std(snn_pcs)),
            },
        },
    }
    out_json = (
        PROJECT_ROOT / "outputs" / "eval_local"
        / "synthetic_validation_stats.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(stats_out, f, indent=2)

    print()
    print("=" * 60)
    print("SYNTHETIC POPULATION VALIDATION")
    print("=" * 60)
    for label, k in ks_results.items():
        print(
            f"  {label:>14}: KS Mamba D={k['mamba_vs_gt']['D']:.3f}  "
            f"SNN D={k['snn_vs_gt']['D']:.3f}"
        )
    print(
        f"  pairwise corr recovery: "
        f"Mamba mean r={np.mean(mamba_pcs):.3f}, "
        f"SNN mean r={np.mean(snn_pcs):.3f}  "
        f"(per-session, n={len(mamba_pcs)})"
    )
    print(f"  Stats JSON: {out_json}")


if __name__ == "__main__":
    main()
