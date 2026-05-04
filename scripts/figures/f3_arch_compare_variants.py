"""F3 architecture-specific comparison variants.

  v1_heatmap_5panel: GT + 4 architectures as spike-count heatmaps
  v2_rate_traces: population-rate time traces with all architectures
  v3_per_session_scatter: per-session decode accuracy, architecture vs raw
  v4_per_neuron_r_scatter: per-neuron r across architectures on same neurons
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


ARCHS = [
    ("mamba", "Mamba", COLORS["Mamba"]),
    ("transformer", "Transformer", COLORS["Transformer"]),
    ("lru", "LRU", COLORS["LRU"]),
    ("snn_standalone_v12b", "Spiking NN", COLORS["SNN"]),
]


def _load_session(session_idx=10):
    preds_dir = PROJECT_ROOT / "outputs" / "eval_local" / "behavioral_predictions"
    out = {}
    for tag, *_ in ARCHS:
        p = preds_dir / tag / f"session_{session_idx:03d}.npz"
        if p.exists():
            d = np.load(p)
            out[tag] = {
                "pred": d["pred_rates"],
                "gt": d["gt"],
                "m_actual": int(d["m_actual"]),
            }
    return out


def v1_heatmap_5panel(out_dir):
    """GT + 4 architectures as heatmaps."""
    apply_style()
    preds = _load_session(session_idx=10)
    if not preds:
        preds = _load_session(session_idx=0)
    if not preds:
        print("  no predictions found; skip v1")
        return

    any_key = list(preds.keys())[0]
    gt = preds[any_key]["gt"]  # (M, T)
    m_i = preds[any_key]["m_actual"]
    # Window: 15s around middle
    T = gt.shape[1]
    t0 = max(0, T // 2 - 150)
    t1 = t0 + 300
    mean_rate = gt[:m_i].mean(axis=1)
    top = np.argsort(-mean_rate)[:200]

    n_cols = 1 + len(ARCHS)
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(TEXT_WIDTH, 2.2),
        sharey=True,
        constrained_layout=True,
    )
    vmax = float(max(np.percentile(gt[top, t0:t1], 99), 1.0))
    axes[0].imshow(
        gt[top, t0:t1], aspect="auto", cmap="magma",
        vmin=0, vmax=vmax, interpolation="nearest",
    )
    axes[0].set_title("Ground truth", fontsize=9)
    axes[0].set_xlabel("Time (50ms bins)", fontsize=8)
    axes[0].set_ylabel("Neuron (sorted)", fontsize=8)

    rng = np.random.default_rng(0)
    for i, (tag, label, color) in enumerate(ARCHS, start=1):
        d = preds.get(tag)
        if d is None:
            axes[i].axis("off")
            continue
        rates = d["pred"][top, t0:t1]
        samp = rng.poisson(np.clip(rates, 0, 50)).astype(np.float32)
        axes[i].imshow(
            samp, aspect="auto", cmap="magma",
            vmin=0, vmax=vmax, interpolation="nearest",
        )
        axes[i].set_title(label, fontsize=9, color=color)
        axes[i].set_xlabel("Time (bins)", fontsize=8)

    for ax in axes:
        ax.tick_params(labelsize=7)
    save_figure(fig, "v1_heatmap_5panel", out_dir=out_dir)
    plt.close(fig)


def v2_rate_traces(out_dir):
    """Population-rate traces with all architectures."""
    apply_style()
    preds = _load_session(session_idx=10)
    if not preds:
        preds = _load_session(session_idx=0)
    if not preds:
        return
    any_key = list(preds.keys())[0]
    gt = preds[any_key]["gt"]
    m_i = preds[any_key]["m_actual"]
    T = gt.shape[1]
    t0, t1 = T // 2 - 200, T // 2 + 200
    x = np.arange(t1 - t0) * 0.05  # seconds

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.0), constrained_layout=True)
    pop_gt = gt[:m_i, t0:t1].mean(axis=0)
    ax.plot(x, pop_gt, color="#222", lw=1.8, label="Ground truth", zorder=5, alpha=0.95)
    for tag, label, color in ARCHS:
        d = preds.get(tag)
        if d is None:
            continue
        pop = d["pred"][:m_i, t0:t1].mean(axis=0)
        ax.plot(x, pop, color=color, lw=1.0, alpha=0.85, label=label)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Mean population rate", fontsize=9)
    ax.set_title(
        "Population-rate trace: ground truth vs architectures (20s)",
        fontsize=9.5, loc="left",
    )
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3, columnspacing=1.0)
    save_figure(fig, "v2_rate_traces", out_dir=out_dir)
    plt.close(fig)


def v3_per_session_scatter(out_dir):
    """Per-session decode accuracy: architecture vs linear baseline."""
    apply_style()
    baseline_p = PROJECT_ROOT / "outputs" / "eval_local" / "behavioral_decode_linear_steinmetz.json"
    baseline = json.load(open(baseline_p))
    base_sess = {s["session_idx"]: s for s in baseline["per_session"]}

    fig, axes = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH, 2.8),
        constrained_layout=True,
    )

    for col, metric_key in enumerate(["resp_acc", "stim_acc"]):
        ax = axes[col]
        for tag, label, color in ARCHS:
            p = PROJECT_ROOT / "outputs" / "eval_local" / f"behavioral_decode_{tag}.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            sessions = d["per_session"]
            xs, ys = [], []
            for s in sessions:
                bs = base_sess.get(s["session_idx"])
                if bs is None:
                    continue
                xs.append(bs[metric_key])
                ys.append(s[metric_key])
            ax.scatter(
                xs, ys, color=color, s=18, alpha=0.8,
                edgecolors="white", linewidths=0.4, label=label,
            )
        # Diagonal
        lo = 0.0
        hi = 1.0
        ax.plot([lo, hi], [lo, hi], color="#888", ls="--", lw=0.7, zorder=0)
        ax.set_xlabel("Linear / raw counts", fontsize=8.5)
        ax.set_ylabel(
            "Architecture predictions" if col == 0 else "",
            fontsize=8.5,
        )
        ax.set_title(
            "Response 3-class\n(per-session, bin-level)"
            if col == 0
            else "Stimulus 16-class\n(per-session, bin-level)",
            fontsize=9,
        )
        # Tight lims
        ax.set_xlim(lo, 0.85)
        ax.set_ylim(lo, 0.85)
        if col == 0:
            ax.legend(fontsize=7, frameon=False, loc="lower right")
    save_figure(fig, "v3_per_session_scatter", out_dir=out_dir)
    plt.close(fig)


def v4_per_neuron_r_scatter(out_dir):
    """Per-neuron r scatter: architecture A vs architecture B."""
    apply_style()
    results_dir = PROJECT_ROOT / "outputs" / "eval_local"
    # Get per-neuron r data from the combined eval files (1L vs 2L)
    files = {
        "1L SNN": results_dir / "multihead_1l_v3_full.json",
        "2L SNN": results_dir / "multihead_2l_v3_full.json",
        "Mamba": results_dir / "mamba_combined_corrected.json",
    }
    data = {}
    for name, p in files.items():
        if p.exists():
            data[name] = json.load(open(p))
    if len(data) < 2:
        print("  not enough per-neuron data; skip v4")
        return

    # Concatenate per-neuron r across sessions for each model (Steinmetz only to be fair)
    def collect(d):
        rs = []
        for s in d["per_session"]:
            if s.get("source") == "ibl":
                continue
            per = s.get("per_neuron_r")
            if per is not None:
                rs.extend(per)
        return np.array(rs)

    mamba_r = collect(data.get("Mamba", {"per_session": []}))
    snn1_r = collect(data.get("1L SNN", {"per_session": []}))
    n = min(len(mamba_r), len(snn1_r))
    if n < 100:
        print("  no per_neuron_r arrays; skip v4")
        return

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.6, 2.6), constrained_layout=True)
    ax.scatter(
        mamba_r[:n], snn1_r[:n], s=6, alpha=0.3,
        color=COLORS["SNN"], edgecolors="none",
    )
    mx = max(mamba_r[:n].max(), snn1_r[:n].max())
    ax.plot([0, mx], [0, mx], color="#888", ls="--", lw=0.7)
    ax.set_xlabel("Mamba per-neuron $r$", fontsize=9)
    ax.set_ylabel("1L SNN per-neuron $r$", fontsize=9)
    ax.set_title(
        f"Per-neuron correlation ({n:,} neurons, Steinmetz)",
        fontsize=9,
    )
    # Density-of-points effect
    save_figure(fig, "v4_per_neuron_r_scatter", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = (
        PROJECT_ROOT / "docs" / "neurips_neurocog"
        / "figure_candidates" / "F3_arch_compare"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for fn in [v1_heatmap_5panel, v2_rate_traces, v3_per_session_scatter, v4_per_neuron_r_scatter]:
        print(f"Generating {fn.__name__}...")
        fn(out_dir)
    print(f"Wrote {len(list(out_dir.glob('*.png')))} PNGs to {out_dir}")
