"""NEDS-inspired figure mockups (NOT for paper).

Three mockups exploring whether NEDS visual motifs are worth adapting:

  Mockup 1 — per-session scatter
      Mamba (3-seed mean) vs H=10 sum baseline, one point per session,
      y=x diagonal, three panels (response / stim 16 / side 3).
      NEDS Fig 3 motif.

  Mockup 2 — color-palette demo
      Same cross-arch trial-vote data rendered in three palette options.
      Lets us see whether a different palette would feel more
      "NEDS-like" without restyling the figures themselves.

  Mockup 3 — multi-panel restyle
      Composite NEDS Fig 2-style: top bar chart, bottom-left per-session
      scatter, bottom-right GT-vs-Mamba snippet. The "what would Fig 2
      look like as a four-panel" question.

Outputs land in docs/neurips_neurocog/figures/mockups/neds_*.png — these
are scratch renders, not paper figures.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
EVAL = ROOT / "outputs" / "eval_local"
OUT = ROOT / "docs" / "neurips_neurocog" / "figures" / "mockups"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_per_session(path):
    d = json.load(open(path))
    return {r["session_idx"]: r for r in d.get("per_session", [])}


def mamba_3seed_per_session():
    """Mean across 3 Mamba seeds, per session, per target."""
    seeds = [
        load_per_session(EVAL / "behavioral_decode_mamba.json"),
        load_per_session(EVAL / "behavioral_decode_mamba_s1.json"),
        load_per_session(EVAL / "behavioral_decode_mamba_s2.json"),
    ]
    sess_ids = sorted(set().union(*[set(s) for s in seeds]))
    rows = []
    for sid in sess_ids:
        if not all(sid in s for s in seeds):
            continue
        rec = {"session_idx": sid}
        for tgt in ("resp_acc", "stim_acc", "side_acc"):
            rec[tgt] = float(np.mean([s[sid][tgt] for s in seeds]))
        rec["n_eval_bins"] = int(seeds[0][sid].get("n_eval_bins", 0))
        rows.append(rec)
    return rows


# ---------------------------------------------------------------------------
# Mockup 1 — per-session scatter, 3 panels (resp / stim 16 / side 3)
# ---------------------------------------------------------------------------
def mockup_1_per_session_scatter():
    mamba = {r["session_idx"]: r for r in mamba_3seed_per_session()}
    base = load_per_session(EVAL / "behavioral_baseline_h10.json")
    sess_ids = sorted(set(mamba) & set(base))

    targets = [
        ("resp_acc", "Response (3-class)", 1 / 3),
        ("stim_acc", "Stimulus contrast (16-class)", 1 / 16),
        ("side_acc", "Stimulus side (3-class)", 1 / 3),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    # NEDS-inspired palette — muted blues + a single warm accent
    colour_above = "#2c7bb6"  # blue (Mamba > baseline)
    colour_below = "#d7191c"  # red (baseline > Mamba)

    for ax, (key, title, chance) in zip(axes, targets):
        x = np.array([base[s][key] for s in sess_ids])
        y = np.array([mamba[s][key] for s in sess_ids])
        win = y > x
        # y=x reference line
        lo = min(x.min(), y.min(), chance) - 0.03
        hi = max(x.max(), y.max()) + 0.03
        ax.plot([lo, hi], [lo, hi], color="0.5", linestyle="--",
                linewidth=1.0, zorder=1)
        # Chance line (vertical + horizontal)
        ax.axvline(chance, color="0.85", linewidth=0.6, zorder=0)
        ax.axhline(chance, color="0.85", linewidth=0.6, zorder=0)
        # Points
        ax.scatter(
            x[win], y[win], s=24, c=colour_above, edgecolor="white",
            linewidth=0.4, zorder=3,
            label=f"Mamba wins ({int(win.sum())}/{len(win)})",
        )
        ax.scatter(
            x[~win], y[~win], s=24, c=colour_below, edgecolor="white",
            linewidth=0.4, zorder=3,
            label=f"Baseline wins ({int((~win).sum())}/{len(win)})",
        )
        # Improvement annotation
        delta = (y - x).mean() * 100
        ax.text(
            0.04, 0.96, f"Mean $\\Delta$ = {delta:+.1f} pp",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6),
        )
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel("H=10 sum baseline (bin acc)")
        ax.set_ylabel("Mamba 3-seed (bin acc)")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    fig.suptitle(
        "Mockup 1 — Per-session matched-context comparison "
        "(Steinmetz 39, NEDS Fig 3 motif)",
        fontsize=11, y=1.04,
    )
    fig.savefig(OUT / "neds_mockup_1_per_session_scatter.png", dpi=160,
                bbox_inches="tight")
    fig.savefig(OUT / "neds_mockup_1_per_session_scatter.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'neds_mockup_1_per_session_scatter.png'}")


# ---------------------------------------------------------------------------
# Mockup 2 — color palette demo (3 options, same bars)
# ---------------------------------------------------------------------------
def mockup_2_palette_demo():
    # Trial-vote response across architectures (canonical numbers)
    archs = ["1-bin LR", "H=10 sum", "H=10 flat", "Mamba", "Transformer",
             "LRU", "NDT2-style", "SNN"]
    resp = [72.1, 69.6, 71.3, 75.7, 75.9, 75.5, 73.0, 67.0]
    sem = [0.0, 0.0, 0.0, 0.2, 0.2, 0.3, 0.0, 2.7]
    is_baseline = [True, True, True, False, False, False, False, False]

    palettes = {
        "Current (per-arch categorical)": {
            "1-bin LR": "#999999", "H=10 sum": "#777777", "H=10 flat": "#555555",
            "Mamba": "#1f77b4", "Transformer": "#ff7f0e", "LRU": "#2ca02c",
            "NDT2-style": "#d62728", "SNN": "#9467bd",
        },
        "NEDS-inspired (muted + accent)": {
            "1-bin LR": "#cccccc", "H=10 sum": "#bbbbbb", "H=10 flat": "#aaaaaa",
            "Mamba": "#2c7bb6", "Transformer": "#abd9e9", "LRU": "#abd9e9",
            "NDT2-style": "#abd9e9", "SNN": "#fdae61",
        },
        "ColorBrewer Set2 (qualitative)": {
            "1-bin LR": "#cccccc", "H=10 sum": "#bbbbbb", "H=10 flat": "#aaaaaa",
            "Mamba": "#66c2a5", "Transformer": "#fc8d62", "LRU": "#8da0cb",
            "NDT2-style": "#e78ac3", "SNN": "#a6d854",
        },
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), constrained_layout=True)
    for ax, (pname, pmap) in zip(axes, palettes.items()):
        colors = [pmap[a] for a in archs]
        bars = ax.bar(range(len(archs)), resp, yerr=sem, color=colors,
                      edgecolor="0.3", linewidth=0.6, capsize=3)
        # Annotate
        for i, (b, v, s) in enumerate(zip(bars, resp, sem)):
            label = f"{v:.1f}"
            if s > 0:
                label += f"$\\pm${s:.1f}"
            ax.text(b.get_x() + b.get_width() / 2, v + 1, label,
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(archs)))
        ax.set_xticklabels(archs, rotation=30, ha="right", fontsize=9)
        ax.axhline(33.3, color="0.6", linestyle="--", linewidth=0.7, zorder=0)
        ax.text(0.02, 33.3 / 90, "chance 33.3%", transform=ax.transAxes,
                fontsize=7, color="0.5", va="bottom")
        ax.set_ylim(60, 82)
        ax.set_ylabel("Response trial-vote (%)")
        ax.set_title(pname, fontsize=10)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Mockup 2 — Same data, three palette options "
        "(no figure restyling, just colors)",
        fontsize=11, y=1.05,
    )
    fig.savefig(OUT / "neds_mockup_2_palette_demo.png", dpi=160,
                bbox_inches="tight")
    fig.savefig(OUT / "neds_mockup_2_palette_demo.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'neds_mockup_2_palette_demo.png'}")


# ---------------------------------------------------------------------------
# Mockup 3 — multi-panel restyle (top bars + bottom-left scatter +
# bottom-right GT-vs-pred snippet)
# ---------------------------------------------------------------------------
def mockup_3_multi_panel():
    # Data: same bars as mockup 2
    archs = ["1-bin LR", "H=10 sum", "H=10 flat", "Mamba", "Transformer",
             "LRU", "NDT2-style", "SNN"]
    resp = [72.1, 69.6, 71.3, 75.7, 75.9, 75.5, 73.0, 67.0]
    sem = [0.0, 0.0, 0.0, 0.2, 0.2, 0.3, 0.0, 2.7]

    mamba = {r["session_idx"]: r for r in mamba_3seed_per_session()}
    base = load_per_session(EVAL / "behavioral_baseline_h10.json")
    sess_ids = sorted(set(mamba) & set(base))
    x_resp = np.array([base[s]["resp_acc"] for s in sess_ids])
    y_resp = np.array([mamba[s]["resp_acc"] for s in sess_ids])

    # GT vs pred snippet — pick a high-r session, take a 200-bin slice,
    # population sum (sum over neurons)
    pred_path = (
        ROOT / "outputs" / "eval_local" / "behavioral_predictions" / "mamba"
        / "session_010.npz"
    )
    arr = np.load(pred_path)
    pop_gt = arr["gt"].sum(axis=0)
    pop_pred = arr["pred_rates"].sum(axis=0)
    # Pick a window with action — find the bin range with highest variance in GT
    win = 250
    var = np.array([
        pop_gt[i:i + win].var() for i in range(0, len(pop_gt) - win, win)
    ])
    start = int(np.argmax(var) * win)
    end = start + win
    t = np.arange(start, end) * 0.05  # 50 ms bins -> seconds

    # Layout: 2 rows. Row 1 = bars (full width). Row 2 = scatter | gt-vs-pred
    fig = plt.figure(figsize=(11.5, 6.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax_bars = fig.add_subplot(gs[0, :])
    ax_scat = fig.add_subplot(gs[1, 0])
    ax_trace = fig.add_subplot(gs[1, 1])

    # --- Bars (NEDS-style: muted greys for baselines, accent blue for Mamba) ---
    base_color = "#bbbbbb"
    mamba_color = "#2c7bb6"
    other_color = "#abd9e9"
    accent_warn = "#fdae61"
    colors = [base_color] * 3 + [mamba_color, other_color, other_color,
                                  other_color, accent_warn]
    bars = ax_bars.bar(range(len(archs)), resp, yerr=sem, color=colors,
                       edgecolor="0.3", linewidth=0.5, capsize=3)
    for b, v, s in zip(bars, resp, sem):
        lab = f"{v:.1f}" + (f"$\\pm${s:.1f}" if s > 0 else "")
        ax_bars.text(b.get_x() + b.get_width() / 2, v + 0.5, lab,
                     ha="center", va="bottom", fontsize=8)
    ax_bars.set_xticks(range(len(archs)))
    ax_bars.set_xticklabels(archs, fontsize=9)
    ax_bars.set_ylabel("Response trial-vote (%)")
    ax_bars.set_ylim(60, 82)
    ax_bars.axhline(33.3, color="0.6", linestyle=":", linewidth=0.6)
    ax_bars.spines["top"].set_visible(False)
    ax_bars.spines["right"].set_visible(False)
    ax_bars.set_title(
        "(a) Aggregate trial-vote across architectures (Steinmetz 39, "
        "1,994 held-out trials)",
        fontsize=10, loc="left",
    )

    # --- Scatter (NEDS Fig 3 motif) ---
    win_mask = y_resp > x_resp
    lo = min(x_resp.min(), y_resp.min()) - 0.03
    hi = max(x_resp.max(), y_resp.max()) + 0.03
    ax_scat.plot([lo, hi], [lo, hi], color="0.5", linestyle="--", linewidth=1)
    ax_scat.scatter(x_resp[win_mask], y_resp[win_mask], s=28,
                    c=mamba_color, edgecolor="white", linewidth=0.4,
                    label=f"Mamba wins ({int(win_mask.sum())}/{len(win_mask)})")
    ax_scat.scatter(x_resp[~win_mask], y_resp[~win_mask], s=28,
                    c=accent_warn, edgecolor="white", linewidth=0.4,
                    label=f"Baseline wins ({int((~win_mask).sum())}/{len(win_mask)})")
    delta = (y_resp - x_resp).mean() * 100
    ax_scat.text(
        0.04, 0.96, f"Mean $\\Delta$ = {delta:+.1f} pp",
        transform=ax_scat.transAxes, va="top", ha="left", fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6),
    )
    ax_scat.set_xlim(lo, hi); ax_scat.set_ylim(lo, hi)
    ax_scat.set_aspect("equal")
    ax_scat.set_xlabel("H=10 sum baseline (bin acc)")
    ax_scat.set_ylabel("Mamba 3-seed (bin acc)")
    ax_scat.set_title("(b) Per-session response decoding", fontsize=10, loc="left")
    ax_scat.legend(loc="lower right", fontsize=8, frameon=False)
    ax_scat.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    # --- GT vs pred trace ---
    ax_trace.plot(t, pop_gt[start:end], color="0.3", linewidth=1.2,
                  label="Recorded population rate")
    ax_trace.plot(t, pop_pred[start:end], color=mamba_color, linewidth=1.2,
                  label="Mamba forecast")
    ax_trace.set_xlabel("Time (s)")
    ax_trace.set_ylabel("$\\sum$ spikes / 50 ms bin")
    ax_trace.set_title(
        "(c) Population-rate forecast (session 010, 12.5 s window)",
        fontsize=10, loc="left",
    )
    ax_trace.legend(loc="upper right", fontsize=8, frameon=False)
    ax_trace.spines["top"].set_visible(False)
    ax_trace.spines["right"].set_visible(False)

    fig.suptitle(
        "Mockup 3 — Multi-panel restyle (NEDS Fig 2 motif): "
        "aggregate + per-session + GT-vs-pred",
        fontsize=11, y=1.02,
    )
    fig.savefig(OUT / "neds_mockup_3_multi_panel.png", dpi=160,
                bbox_inches="tight")
    fig.savefig(OUT / "neds_mockup_3_multi_panel.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'neds_mockup_3_multi_panel.png'}")


if __name__ == "__main__":
    print("Rendering NEDS-inspired mockups...")
    mockup_1_per_session_scatter()
    mockup_2_palette_demo()
    mockup_3_multi_panel()
    print("Done.")
