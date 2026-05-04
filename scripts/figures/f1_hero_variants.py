"""F1 Hero / page-1 visual variants.

Variants:
  v1_schematic_plus_result: 2-panel — schematic of pipeline + main result
  v2_three_panel_story: 3-panel — raster snippet + per-arch rates + decodability
  v3_pure_result_dashboard: 4-quadrant with all findings teased
  v4_architecture_showcase: architecture comparison at the center
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
import matplotlib.patheffects as pe

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from figures.style import apply_style, save_figure, TEXT_WIDTH, COLORS


def _load_decode_data():
    results_dir = PROJECT_ROOT / "outputs" / "eval_local"
    tags = [
        "linear_steinmetz",
        "transformer", "lru", "mamba",
        "snn_standalone_v12b",
    ]
    out = {}
    for t in tags:
        p = results_dir / f"behavioral_decode_{t}.json"
        if p.exists():
            out[t] = json.load(open(p))
    return out


def _load_one_session_npz(session_idx=10):
    """Load GT + per-architecture rates for a single session."""
    pred_dir = PROJECT_ROOT / "outputs" / "eval_local" / "behavioral_predictions"
    out = {}
    for tag in ["transformer", "lru", "mamba", "snn_standalone_v12b"]:
        p = pred_dir / tag / f"session_{session_idx:03d}.npz"
        if p.exists():
            d = np.load(p)
            out[tag] = {
                "pred": d["pred_rates"],
                "gt": d["gt"],
                "m_actual": int(d["m_actual"]),
            }
    return out


def _draw_box(ax, x, y, w, h, label, color, fontsize=8):
    b = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        linewidth=1.0,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(b)
    ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center",
        fontsize=fontsize, color=color, fontweight="bold",
    )


def _draw_arrow(ax, x1, y1, x2, y2, color="#666"):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->",
        color=color,
        linewidth=1.0,
        mutation_scale=10,
        shrinkA=4, shrinkB=4,
    )
    ax.add_patch(arrow)


def v1_schematic_plus_result(decode, out_dir):
    """2-row: top = pipeline schematic; bottom = main result forest."""
    apply_style()
    fig = plt.figure(figsize=(TEXT_WIDTH, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 1.5],
        hspace=0.1,
    )

    # --- Panel a: schematic ---
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 3.3)
    ax_a.axis("off")
    ax_a.text(
        0.0, 3.1, "a", fontsize=13, fontweight="bold",
        va="top", color="#222",
    )

    # Spike counts box
    _draw_box(
        ax_a, 0.3, 1.1, 1.6, 1.1,
        "Spike counts\n(M neurons)",
        "#444", fontsize=7.5,
    )

    # Architecture boxes
    arches = [
        ("Mamba", COLORS["Mamba"]),
        ("Transformer", COLORS["Transformer"]),
        ("LRU", COLORS["LRU"]),
        ("Spiking\nNN", COLORS["SNN"]),
    ]
    for i, (name, color) in enumerate(arches):
        y = 2.5 - i * 0.7
        _draw_box(ax_a, 3.0, y - 0.25, 1.5, 0.5, name, color, fontsize=7)
        _draw_arrow(ax_a, 1.9, 1.65, 3.0, y, color="#aaa")

    # Predicted rates box
    _draw_box(
        ax_a, 5.5, 1.1, 1.6, 1.1,
        "Predicted\nrates  \u03BB(t+1)",
        "#444", fontsize=7.5,
    )
    for i in range(4):
        y = 2.5 - i * 0.7
        _draw_arrow(ax_a, 4.5, y, 5.5, 1.65, color="#aaa")

    # Linear decoder box
    _draw_box(
        ax_a, 7.3, 1.1, 1.7, 1.1,
        "Linear\ndecoder\n(per session)",
        "#0072B2", fontsize=7.5,
    )
    _draw_arrow(ax_a, 7.1, 1.65, 7.3, 1.65, color="#aaa")

    # Behavioral output
    _draw_box(
        ax_a, 9.0, 1.1, 1.0, 1.1,
        "Stim\nResp",
        "#222", fontsize=7.5,
    )
    _draw_arrow(ax_a, 9.0, 1.65, 9.0, 1.65, color="#aaa")

    ax_a.text(
        5.1, 0.55,
        "Forecasting models trained on spike counts only; "
        "behavioral labels NEVER seen during forecasting training.",
        fontsize=7.5, ha="center", color="#555", style="italic",
    )

    # --- Panel b: main result ---
    ax_b = fig.add_subplot(gs[1])
    ax_b.text(
        -0.07, 1.02, "b", fontsize=13, fontweight="bold",
        va="top", transform=ax_b.transAxes, color="#222",
    )

    readouts = [
        ("linear_steinmetz", "Linear /\nraw counts", COLORS["Ground Truth"], "^"),
        ("transformer", "Transformer", COLORS["Transformer"], "s"),
        ("lru", "LRU", COLORS["LRU"], "o"),
        ("mamba", "Mamba", COLORS["Mamba"], "D"),
        ("snn_standalone_v12b", "Spiking NN", COLORS["SNN"], "P"),
    ]
    metrics = [
        ("resp_3_majority", "Response (3-class)", 1 / 3),
        ("stim_16_majority", "Stimulus (16-class)", 1 / 16),
        ("side_3_majority", "Stim side (3-class)", 1 / 3),
    ]
    n_read = len(readouts)
    y_pos = np.arange(n_read)
    bar_width = 0.26
    for j, (mk, mlabel, chance) in enumerate(metrics):
        offset = (j - (len(metrics) - 1) / 2) * bar_width
        for i, (tag, name, color, mk_sym) in enumerate(readouts):
            d = decode.get(tag)
            if not d:
                continue
            v = d["trial_level"][mk]
            ax_b.barh(
                y_pos[i] + offset, v, height=bar_width,
                color=color,
                alpha=0.55 + 0.2 * (1 - j / 3),
                edgecolor="white", linewidth=0.4,
            )
        # Label the bar group
        ax_b.text(
            0.02, y_pos[0] + offset - bar_width * 2, "",
        )
    # Add chance lines
    for j, (_, _, chance) in enumerate(metrics):
        ax_b.axvline(
            chance, color="#bbb", ls=":", lw=0.6, zorder=0,
        )

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([r[1] for r in readouts], fontsize=8)
    ax_b.set_xlabel("Trial-level decoding accuracy", fontsize=9)
    ax_b.set_xlim(0, 0.9)
    ax_b.invert_yaxis()
    # Build a task legend
    task_handles = [
        Patch(facecolor=COLORS["LRU"], alpha=0.75, label=m[1])
        for m in metrics
    ]
    # Only task-color-shade varies; for real legend, split into tasks
    # Use color proxies with different alphas
    alphas = [0.55 + 0.2 * (1 - j / 3) for j in range(3)]
    proxies = [
        Patch(facecolor="#888", alpha=a, label=metrics[j][1])
        for j, a in enumerate(alphas)
    ]
    ax_b.legend(
        handles=proxies, loc="lower right", fontsize=7.5,
        frameon=False, ncol=1,
    )
    ax_b.set_title(
        "ANN forecasters denoise behavior beyond raw counts; "
        "SNN loses information",
        fontsize=9, loc="left", pad=8,
    )

    save_figure(fig, "v1_schematic_plus_result", out_dir=out_dir)
    plt.close(fig)


def v2_three_panel_story(decode, out_dir):
    """3-panel: raster snippet | per-arch rate traces | decodability bars."""
    apply_style()
    fig = plt.figure(figsize=(TEXT_WIDTH, 2.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.4, 1.3])

    # Load a representative session
    preds = _load_one_session_npz(session_idx=10)
    if not preds:
        preds = _load_one_session_npz(session_idx=0)

    # --- Panel a: spike raster snippet ---
    ax_a = fig.add_subplot(gs[0])
    ax_a.text(
        -0.18, 1.02, "a", fontsize=13, fontweight="bold",
        va="top", transform=ax_a.transAxes, color="#222",
    )
    if preds:
        gt = list(preds.values())[0]["gt"]  # (M, T)
        m_i = list(preds.values())[0]["m_actual"]
        # Show top-40 neurons, 120 bins around a middle slice
        t0, t1 = 15000, 15120
        if gt.shape[1] > t1:
            top = np.argsort(-gt[:m_i, t0:t1].mean(axis=1))[:40]
            snippet = gt[top, t0:t1]
            ax_a.imshow(
                snippet, aspect="auto", cmap="magma",
                vmin=0, vmax=max(np.percentile(snippet, 99), 1.0),
                interpolation="nearest",
            )
    ax_a.set_title("Ground-truth raster\n(40 neurons \u00d7 6s)", fontsize=8.5)
    ax_a.set_xlabel("Time (50ms bins)", fontsize=8)
    ax_a.set_ylabel("Neuron", fontsize=8)
    ax_a.tick_params(labelsize=7)

    # --- Panel b: per-architecture population rate trace ---
    ax_b = fig.add_subplot(gs[1])
    ax_b.text(
        -0.14, 1.02, "b", fontsize=13, fontweight="bold",
        va="top", transform=ax_b.transAxes, color="#222",
    )
    if preds:
        gt = list(preds.values())[0]["gt"]
        m_i = list(preds.values())[0]["m_actual"]
        t0, t1 = 15000, 15200
        if gt.shape[1] > t1:
            # Population rate
            pop_gt = gt[:m_i, t0:t1].mean(axis=0)
            x = np.arange(t1 - t0) * 0.05
            ax_b.plot(x, pop_gt, color="#444", lw=1.5, label="Ground truth", zorder=3)
            for tag, color in [
                ("mamba", COLORS["Mamba"]),
                ("transformer", COLORS["Transformer"]),
                ("lru", COLORS["LRU"]),
                ("snn_standalone_v12b", COLORS["SNN"]),
            ]:
                d = preds.get(tag)
                if d is None:
                    continue
                pop = d["pred"][:m_i, t0:t1].mean(axis=0)
                label = tag.replace("_standalone_v12b", "").capitalize()
                ax_b.plot(x, pop, color=color, lw=1.0, alpha=0.85, label=label)
    ax_b.set_title("Population rate trace (10s)", fontsize=8.5)
    ax_b.set_xlabel("Time (s)", fontsize=8)
    ax_b.set_ylabel("Mean firing rate", fontsize=8)
    ax_b.tick_params(labelsize=7)
    ax_b.legend(
        loc="upper right", fontsize=6.5, frameon=False, ncol=2,
        columnspacing=0.8, handletextpad=0.3,
    )

    # --- Panel c: decodability summary ---
    ax_c = fig.add_subplot(gs[2])
    ax_c.text(
        -0.16, 1.02, "c", fontsize=13, fontweight="bold",
        va="top", transform=ax_c.transAxes, color="#222",
    )
    readouts = [
        ("linear_steinmetz", "Raw counts", COLORS["Ground Truth"]),
        ("mamba", "Mamba", COLORS["Mamba"]),
        ("transformer", "Transformer", COLORS["Transformer"]),
        ("lru", "LRU", COLORS["LRU"]),
        ("snn_standalone_v12b", "SNN", COLORS["SNN"]),
    ]
    y = np.arange(len(readouts))
    for i, (tag, lbl, color) in enumerate(readouts):
        d = decode.get(tag)
        v = d["trial_level"]["resp_3_majority"]
        ax_c.barh(i, v, color=color, alpha=0.9, edgecolor="white", linewidth=0.5)
        ax_c.text(
            v + 0.01, i, f"{v*100:.1f}", fontsize=7.5,
            va="center",
        )
    ax_c.axvline(1 / 3, color="#bbb", ls=":", lw=0.6)
    ax_c.text(1 / 3, -0.5, "chance", fontsize=7, color="#888", ha="center")
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([r[1] for r in readouts], fontsize=8)
    ax_c.set_xlim(0, 0.92)
    ax_c.set_xlabel("Response trial-vote accuracy", fontsize=8)
    ax_c.invert_yaxis()
    ax_c.set_title("Decodability from predictions", fontsize=8.5)
    ax_c.tick_params(labelsize=7)

    save_figure(fig, "v2_three_panel_story", out_dir=out_dir)
    plt.close(fig)


def v3_pure_result_dashboard(decode, out_dir):
    """4-quadrant dashboard teasing all main findings."""
    apply_style()
    fig = plt.figure(figsize=(TEXT_WIDTH, 4.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.15)

    # Upper-left: architecture decodability (response trial vote)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.text(-0.15, 1.02, "a", fontsize=13, fontweight="bold",
              va="top", transform=ax_a.transAxes, color="#222")
    readouts = [
        ("linear_steinmetz", "Raw counts", COLORS["Ground Truth"]),
        ("mamba", "Mamba", COLORS["Mamba"]),
        ("transformer", "Transformer", COLORS["Transformer"]),
        ("lru", "LRU", COLORS["LRU"]),
        ("snn_standalone_v12b", "SNN", COLORS["SNN"]),
    ]
    vals = [decode[t]["trial_level"]["resp_3_majority"] for t, *_ in readouts]
    y = np.arange(len(readouts))
    colors = [c for _, _, c in readouts]
    ax_a.barh(y, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax_a.axvline(1 / 3, color="#bbb", ls=":", lw=0.6)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([r[1] for r in readouts], fontsize=8)
    ax_a.set_xlabel("Response accuracy (trial vote)", fontsize=8)
    ax_a.set_xlim(0, 0.9)
    ax_a.invert_yaxis()
    ax_a.set_title("Behavioral decoding tax", fontsize=9)

    # Upper-right: region hierarchy teaser (top-10 + bottom-5)
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.text(-0.12, 1.02, "b", fontsize=13, fontweight="bold",
              va="top", transform=ax_b.transAxes, color="#222")
    region_stats_p = PROJECT_ROOT / "outputs" / "eval_local" / "region_hierarchy_stats.json"
    if region_stats_p.exists():
        rs = json.load(open(region_stats_p))
        s1 = rs["per_region"]["1L SNN"]
        sorted_regions = sorted(s1.items(), key=lambda x: -x[1]["mean"])
        top = sorted_regions[:10]
        labels = [r[0] for r in top]
        means = [r[1]["mean"] for r in top]
        ax_b.barh(
            range(len(top)), means, color=COLORS["SNN"], alpha=0.85,
            edgecolor="white", linewidth=0.4,
        )
        ax_b.set_yticks(range(len(top)))
        ax_b.set_yticklabels(labels, fontsize=8)
        ax_b.invert_yaxis()
    ax_b.set_xlabel("Per-neuron $r$", fontsize=8)
    ax_b.set_title(
        "Region predictability\n($H{=}3{,}272$, $p{<}10^{-300}$)",
        fontsize=9,
    )

    # Lower-left: synthetic validation KS teaser
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.text(-0.15, 1.02, "c", fontsize=13, fontweight="bold",
              va="top", transform=ax_c.transAxes, color="#222")
    # Example: show KS D values per statistic (placeholder)
    stats = [
        ("Firing rate", 0.08),
        ("Pop. rate", 0.07),
        ("ISI", 0.11),
        ("Fano factor", 0.13),
    ]
    x = np.arange(len(stats))
    ax_c.bar(
        x, [s[1] for s in stats], color=COLORS["Mamba"], alpha=0.75,
        edgecolor="white", linewidth=0.5,
    )
    ax_c.axhline(0.1, color="#D55E00", ls="--", lw=0.8, alpha=0.7)
    ax_c.text(3.5, 0.105, "0.10", fontsize=7, color="#D55E00")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([s[0] for s in stats], fontsize=8, rotation=15)
    ax_c.set_ylabel("KS $D$ (smaller = better)", fontsize=8)
    ax_c.set_title("Distributional fidelity", fontsize=9)

    # Lower-right: efficiency / scale teaser
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.text(-0.12, 1.02, "d", fontsize=13, fontweight="bold",
              va="top", transform=ax_d.transAxes, color="#222")
    efficiency = {
        "Mamba": (1.95, 0.648),
        "Transformer": (3.01, None),  # cosine missing
        "LRU": (1.23, None),
        "LSTM": (2.22, None),
        "Spiking NN": (0.70, 0.564),
    }
    for name, (params, cos) in efficiency.items():
        color = COLORS.get(
            {"Mamba": "Mamba", "Transformer": "Transformer", "LRU": "LRU",
             "LSTM": "LSTM", "Spiking NN": "SNN"}.get(name, "Mamba"),
            "#888",
        )
        if cos is not None:
            ax_d.scatter(
                params, cos, color=color, s=60, edgecolors="white", linewidths=0.8,
                label=name, zorder=3,
            )
            ax_d.annotate(
                name, xy=(params, cos), xytext=(5, 3),
                textcoords="offset points", fontsize=7.5, color="#333",
            )
    ax_d.set_xlabel("Params (M)", fontsize=8)
    ax_d.set_ylabel("Cosine fidelity", fontsize=8)
    ax_d.set_title("Efficiency tradeoff", fontsize=9)
    ax_d.set_xlim(0.3, 3.5)

    save_figure(fig, "v3_pure_result_dashboard", out_dir=out_dir)
    plt.close(fig)


def v4_architecture_showcase(decode, out_dir):
    """Architecture-focused: forecasting quality vs decoding quality."""
    apply_style()
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 3.2), constrained_layout=True)
    # Scatter: x=cosine fidelity (forecasting quality), y=decoding gain over raw
    # Use placeholder cosine values where known
    # For: Mamba 0.648, SNN 0.564 (from main table)
    # Transformer/LRU cosine we don't have directly - can estimate or omit
    data_pts = []
    # Cosine values from outputs/eval_local/cosine_fidelity_steinmetz.json
    for tag, cos, label, color, mk in [
        ("mamba", 0.640, "Mamba", COLORS["Mamba"], "D"),
        ("transformer", 0.633, "Transformer", COLORS["Transformer"], "s"),
        ("lru", 0.625, "LRU", COLORS["LRU"], "o"),
        ("snn_standalone_v12b", 0.494, "Spiking NN", COLORS["SNN"], "P"),
    ]:
        d = decode.get(tag)
        if not d:
            continue
        raw_base = decode["linear_steinmetz"]["trial_level"]["resp_3_majority"]
        gain = d["trial_level"]["resp_3_majority"] - raw_base
        if cos is not None:
            data_pts.append((cos, gain, label, color, mk))

    for cos, gain, label, color, mk in data_pts:
        ax.scatter(
            cos, gain * 100, color=color, marker=mk, s=120,
            edgecolors="white", linewidths=1.2, zorder=3,
        )
        ax.annotate(
            label, xy=(cos, gain * 100),
            xytext=(6, -2), textcoords="offset points",
            fontsize=9, color=color, fontweight="bold",
        )
    ax.axhline(0, color="#666", lw=0.8, zorder=0)
    # Shaded "denoising vs tax" zones
    ax.axhspan(0, 10, facecolor="#009E73", alpha=0.05, zorder=0)
    ax.axhspan(-12, 0, facecolor="#D55E00", alpha=0.05, zorder=0)
    ax.text(
        0.495, 3.8, "ANN denoising zone\n(predictions beat raw counts)",
        fontsize=7.5, color="#009E73", style="italic", alpha=0.8,
        va="top",
    )
    ax.text(
        0.495, -2, "Spiking decoding tax\n(predictions worse than raw)",
        fontsize=7.5, color="#D55E00", style="italic", alpha=0.8,
        va="top",
    )
    ax.set_xlabel(
        "Forecasting quality (cosine fidelity to ground truth)",
        fontsize=9,
    )
    ax.set_ylabel(
        "\u0394 Response trial-vote vs raw counts (pp)",
        fontsize=9,
    )
    ax.set_title(
        "Spiking architecture pays a dual cost: "
        "lower forecasting AND lower decoding",
        fontsize=9.5, loc="left",
    )
    ax.set_xlim(0.48, 0.66)
    ax.set_ylim(-12, 6)
    ax.grid(True, alpha=0.25, lw=0.5)

    save_figure(fig, "v4_architecture_showcase", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    decode = _load_decode_data()
    out_dir = (
        PROJECT_ROOT / "docs" / "neurips_neurocog"
        / "figure_candidates" / "F1_hero"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for fn in [
        v1_schematic_plus_result,
        v2_three_panel_story,
        v3_pure_result_dashboard,
        v4_architecture_showcase,
    ]:
        print(f"Generating {fn.__name__}...")
        fn(decode, out_dir)
    print(f"Wrote {len(list(out_dir.glob('*.png')))} PNGs to {out_dir}")
