"""Hero figure mockup C: example traces + summary.

Two-panel layout:
  (a) Left: stacked rate traces for one example neuron from one session,
      showing GT (gray bars) vs Mamba pred (orange) vs SNN pred (green).
      Behavioral events marked with vertical lines / shading.
  (b) Right: forest plot summary across architectures.

Picks a representative neuron + trial window and overlays all model
predictions to make the "denoising" intuition visible.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from figures.style import apply_style, save_figure, TEXT_WIDTH, COLORS


def _load_pred(tag, sess_idx):
    p = (
        PROJECT_ROOT / "outputs" / "eval_local" / "behavioral_predictions"
        / tag / f"session_{sess_idx:03d}.npz"
    )
    if not p.exists():
        return None, None, None
    arr = np.load(p)
    return arr["pred_rates"], arr["gt"], int(arr.get("split_start_bin", 10))


def main():
    apply_style()

    sess_idx = 4  # session_004 had decent decoding across models
    transformer_pred, gt, sb_t = _load_pred("transformer", sess_idx)
    mamba_pred, _, sb_m = _load_pred("mamba", sess_idx)
    lru_pred, _, sb_l = _load_pred("lru", sess_idx)
    snn_pred, _, sb_s = _load_pred("snn_standalone_v12b", sess_idx)
    if any(x is None for x in [transformer_pred, mamba_pred, lru_pred, snn_pred]):
        raise SystemExit("Missing predictions for example session")

    # Pick a neuron that's active in the trial window
    # Find a window with high activity in GT
    M, T = gt.shape
    pop_rate = gt.sum(axis=0)  # population rate over time
    # Find a 200-bin window with above-average pop rate
    win = 200
    pop_smoothed = np.convolve(pop_rate, np.ones(50) / 50, mode="same")
    start = int(np.argmax(pop_smoothed[100:T - win - 100]) + 100)
    end = start + win
    # Pick a high-rate neuron in this window
    win_rates = gt[:, start:end].mean(axis=1)
    neuron_idx = int(np.argsort(-win_rates)[3])  # 4th most active
    print(
        f"Session {sess_idx}, neuron {neuron_idx} "
        f"(window {start}-{end}, mean rate {win_rates[neuron_idx]:.3f})"
    )

    # Time axis (bins -> seconds at 50ms/bin)
    t = np.arange(win) * 0.05

    # ---- Layout ----
    fig = plt.figure(figsize=(TEXT_WIDTH, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.30)

    # --- Left: trace overlay ---
    ax = fig.add_subplot(gs[0])

    # Ground truth as gray bars
    ax.bar(
        t, gt[neuron_idx, start:end],
        width=0.045, color="#999999", edgecolor="none",
        label="Ground truth (counts)", zorder=1,
    )
    # Predicted rates as smooth lines
    ax.plot(
        t, mamba_pred[neuron_idx, start:end],
        color=COLORS["Mamba"], linewidth=1.4, label="Mamba pred",
        zorder=3,
    )
    ax.plot(
        t, transformer_pred[neuron_idx, start:end],
        color=COLORS["Transformer"], linewidth=1.0, label="Transformer pred",
        alpha=0.9, zorder=3,
    )
    ax.plot(
        t, snn_pred[neuron_idx, start:end],
        color=COLORS["SNN"], linewidth=1.4, label="Spiking NN pred",
        linestyle="--", zorder=3,
    )

    ax.set_xlabel("Time (s)", fontsize=8.5)
    ax.set_ylabel("Spikes / 50 ms bin   /   predicted rate", fontsize=8.5)
    ax.set_title(
        f"Session {sess_idx:03d}, neuron {neuron_idx}: "
        "ANN forecasters smooth count noise; SNN tracks but shrinks",
        fontsize=9, loc="left",
    )
    ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    ax.text(
        -0.10, 1.05, "a", transform=ax.transAxes,
        fontsize=12, fontweight="bold",
    )

    # --- Right: forest plot ---
    rd = PROJECT_ROOT / "outputs" / "eval_local"
    decoders = [
        ("linear_steinmetz", "Raw counts", "#444444"),
        ("transformer", "Transformer", COLORS["Transformer"]),
        ("mamba", "Mamba", COLORS["Mamba"]),
        ("lru", "LRU", COLORS["LRU"]),
        ("snn_standalone_v12b", "Spiking NN", COLORS["SNN"]),
    ]
    rows = []
    for tag, label, color in decoders:
        p = rd / f"behavioral_decode_{tag}.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        sessions = d.get("per_session", [])
        sess_resp = np.array([s.get("resp_acc", np.nan) for s in sessions])
        sess_resp = sess_resp[~np.isnan(sess_resp)]
        agg = d.get("trial_level", {}).get("resp_3_majority", np.nan)
        rows.append((tag, label, color, agg, sess_resp))
    raw_acc = next(r[3] for r in rows if r[0] == "linear_steinmetz")

    ax2 = fig.add_subplot(gs[1])
    n = len(rows)
    yy = np.arange(n)[::-1]

    ax2.add_patch(Rectangle(
        (raw_acc, -0.5), 1 - raw_acc, n,
        facecolor="#d8eed4", alpha=0.30, edgecolor="none", zorder=0,
    ))
    ax2.add_patch(Rectangle(
        (0, -0.5), raw_acc, n,
        facecolor="#f4d4d4", alpha=0.25, edgecolor="none", zorder=0,
    ))
    ax2.axvline(
        raw_acc, color="#444444", linewidth=1.0, linestyle="--", zorder=1,
    )

    for i, (tag, label, color, agg, sess) in enumerate(rows):
        se = float(sess.std() / np.sqrt(len(sess))) if len(sess) > 1 else 0.0
        ax2.errorbar(
            agg, yy[i], xerr=se,
            fmt="o", color=color, markersize=6,
            ecolor=color, elinewidth=0.9, capsize=2.5,
            markeredgewidth=0, zorder=3,
        )

    ax2.set_yticks(yy)
    ax2.set_yticklabels([r[1] for r in rows], fontsize=7.5)
    ax2.set_xlabel("Trial vote\nresp. accuracy", fontsize=8)
    ax2.set_xlim(0.55, 0.82)
    ax2.set_ylim(-0.5, n - 0.5)
    ax2.text(
        -0.20, 1.05, "b", transform=ax2.transAxes,
        fontsize=12, fontweight="bold",
    )
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    fig.suptitle(
        "ANN forecasters smooth single-trial spike noise into a "
        "behavior-aligned rate; spiking forecasters lose the signal",
        fontsize=10, y=1.02,
    )

    out_dir = PROJECT_ROOT / "docs" / "neurips_neurocog" / "figures" / "mockups"
    save_figure(fig, "hero_C_traces", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
