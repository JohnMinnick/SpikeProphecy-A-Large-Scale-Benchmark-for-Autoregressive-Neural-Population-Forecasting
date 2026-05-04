"""Hero figure mockup B: task + pipeline + main result.

Three-panel layout:
  (a) Task strip: mouse / Steinmetz visual-discrimination task / Allen brain map
  (b) Pipeline: spike history -> forecaster -> predicted rates -> linear decoder -> behavior
  (c) Forest plot of trial-vote response accuracy across architectures

Uses simple matplotlib primitives for the task and pipeline schematics.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Rectangle, FancyArrowPatch, Circle, FancyBboxPatch, Wedge, Ellipse,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from figures.style import apply_style, save_figure, TEXT_WIDTH, COLORS


def draw_mouse_icon(ax, x, y, scale=1.0, color="#444444"):
    """Tiny stylized mouse silhouette (top-down)."""
    # Body (oval)
    ax.add_patch(Ellipse(
        (x, y), 0.06 * scale, 0.10 * scale,
        facecolor=color, edgecolor="none",
    ))
    # Head (circle)
    ax.add_patch(Circle(
        (x, y + 0.06 * scale), 0.025 * scale,
        facecolor=color, edgecolor="none",
    ))
    # Ears
    ax.add_patch(Circle(
        (x - 0.018 * scale, y + 0.075 * scale), 0.012 * scale,
        facecolor=color, edgecolor="none",
    ))
    ax.add_patch(Circle(
        (x + 0.018 * scale, y + 0.075 * scale), 0.012 * scale,
        facecolor=color, edgecolor="none",
    ))
    # Tail (line)
    ax.plot(
        [x, x + 0.015 * scale, x],
        [y - 0.05 * scale, y - 0.10 * scale, y - 0.13 * scale],
        color=color, linewidth=0.8, zorder=1,
    )


def draw_grating(ax, x, y, w=0.10, h=0.08, contrast=0.9, side="left"):
    """Vertical sinusoidal grating in a square."""
    # Box
    ax.add_patch(Rectangle(
        (x - w / 2, y - h / 2), w, h,
        facecolor="white", edgecolor="#333333", linewidth=0.6,
    ))
    # Grating bars
    n_bars = 5
    bar_w = w / n_bars
    for i in range(n_bars):
        gray = 0.95 - contrast * 0.7 * (i % 2)
        ax.add_patch(Rectangle(
            (x - w / 2 + i * bar_w, y - h / 2), bar_w, h,
            facecolor=str(gray), edgecolor="none",
        ))
    # Border on top
    ax.add_patch(Rectangle(
        (x - w / 2, y - h / 2), w, h,
        facecolor="none", edgecolor="#333333", linewidth=0.6,
    ))


def draw_wheel_arrow(ax, x, y, direction="left"):
    """Tiny arrow indicating wheel turn direction."""
    # Wheel (circle)
    ax.add_patch(Circle(
        (x, y), 0.025,
        facecolor="white", edgecolor="#333333", linewidth=0.6,
    ))
    # Arrow on wheel
    sign = -1 if direction == "left" else 1
    ax.annotate(
        "",
        xy=(x + sign * 0.018, y - 0.005),
        xytext=(x - sign * 0.012, y + 0.012),
        arrowprops=dict(
            arrowstyle="-|>", linewidth=0.8, color="#333333",
            connectionstyle="arc3,rad=0.3",
        ),
    )


def draw_brain_outline(ax, x, y, scale=1.0):
    """Stylized top-down mouse brain outline with shaded regions."""
    # Cortex outline (kidney-bean-ish)
    from matplotlib.patches import Polygon
    n_pts = 80
    t = np.linspace(0, 2 * np.pi, n_pts)
    # Kidney bean: r = 1 + 0.3 cos(t) - asymmetric
    r = 1.0 + 0.15 * np.cos(t) + 0.10 * np.sin(2 * t)
    bx = x + scale * 0.05 * r * np.cos(t)
    by = y + scale * 0.07 * r * np.sin(t)
    ax.fill(bx, by, facecolor="#f5f5f0", edgecolor="#666666", linewidth=0.6)
    # Highlight a few "regions" with colored patches
    region_pts = [
        (-0.020, 0.025, "#D55E00", "MOs"),     # frontal motor
        (0.020, -0.025, "#0072B2", "VISp"),    # posterior visual
        (-0.005, -0.005, "#009E73", "CA1"),    # hippocampal
    ]
    for dx, dy, color, label in region_pts:
        ax.add_patch(Circle(
            (x + scale * dx, y + scale * dy), 0.012 * scale,
            facecolor=color, edgecolor="none", alpha=0.85,
        ))
        ax.text(
            x + scale * dx, y + scale * dy - 0.022 * scale,
            label, ha="center", va="top",
            fontsize=6.5, color="#222222",
        )


def draw_neuron_grid(ax, x0, y0, w, h, n_rows=8, n_cols=15, density=0.18):
    """Stylized spike raster as gray dots in an n_rows x n_cols grid."""
    rng = np.random.default_rng(0)
    spikes = rng.random((n_rows, n_cols)) < density
    # Ticks for time axis
    for r in range(n_rows):
        for c in range(n_cols):
            if spikes[r, c]:
                ax.add_patch(Rectangle(
                    (x0 + c * w / n_cols + 0.001,
                     y0 + r * h / n_rows + 0.001),
                    w / n_cols * 0.5, h / n_rows * 0.7,
                    facecolor="#333333", edgecolor="none",
                ))
    # Bounding box
    ax.add_patch(Rectangle(
        (x0, y0), w, h,
        facecolor="none", edgecolor="#666666", linewidth=0.5,
    ))


def draw_arch_box(ax, x, y, w, h, label, color, sublabel=None):
    """Stylized architecture box."""
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.005",
        facecolor=color, edgecolor="none", alpha=0.85,
    ))
    ax.text(
        x, y, label, ha="center", va="center",
        fontsize=8, fontweight="bold", color="white",
    )
    if sublabel:
        ax.text(
            x, y - h * 0.7, sublabel, ha="center", va="top",
            fontsize=6.5, color="#444444",
        )


def draw_arrow(ax, x0, x1, y, color="#444444", lw=1.0):
    ax.annotate(
        "",
        xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(
            arrowstyle="-|>", linewidth=lw, color=color,
            mutation_scale=8,
        ),
    )


def main():
    apply_style()

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

    # ---- Figure layout: 2 rows x 3 cols, top row = task+pipeline, bottom = result ----
    fig = plt.figure(figsize=(TEXT_WIDTH, 4.0))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 2.0],
        hspace=0.35,
    )

    # --- Top: task + pipeline as one wide axis ---
    ax_t = fig.add_subplot(gs[0])
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 0.5)
    ax_t.axis("off")

    # Panel a: task strip
    # Mouse + grating + wheel + brain
    draw_mouse_icon(ax_t, 0.05, 0.30, scale=1.5)
    draw_grating(ax_t, 0.13, 0.32, contrast=0.9, side="left")
    draw_grating(ax_t, 0.21, 0.32, contrast=0.5, side="right")
    draw_wheel_arrow(ax_t, 0.27, 0.32, direction="left")
    ax_t.text(
        0.16, 0.10,
        "Steinmetz visual discrimination\n(left/right contrast → wheel turn)",
        ha="center", va="top", fontsize=7, color="#333333",
    )

    draw_brain_outline(ax_t, 0.39, 0.30, scale=1.5)
    ax_t.text(
        0.39, 0.10,
        "39 sessions, ~27K neurons,\n70 Allen CCF regions",
        ha="center", va="top", fontsize=7, color="#333333",
    )

    # Panel b: pipeline (right side)
    py = 0.30  # y-position of pipeline elements
    draw_neuron_grid(ax_t, 0.52, 0.20, 0.08, 0.20, n_rows=6, n_cols=10)
    ax_t.text(0.56, 0.10, "10-bin spike\nhistory", ha="center", va="top",
              fontsize=7, color="#333333")
    draw_arrow(ax_t, 0.60, 0.66, 0.30)
    draw_arch_box(ax_t, 0.71, 0.30, 0.10, 0.10,
                  "Forecaster", "#0072B2",
                  "(Mamba/SNN/...)")
    draw_arrow(ax_t, 0.76, 0.82, 0.30)
    draw_neuron_grid(ax_t, 0.84, 0.20, 0.06, 0.20, n_rows=6, n_cols=8,
                     density=0.35)
    ax_t.text(0.87, 0.10, "predicted\nrates", ha="center", va="top",
              fontsize=7, color="#333333")
    draw_arrow(ax_t, 0.90, 0.97, 0.30, color="#a02929", lw=1.4)

    # Caption labels for top
    ax_t.text(
        0.005, 0.48, "a", fontsize=12, fontweight="bold", va="top",
    )
    ax_t.text(
        0.50, 0.48, "b", fontsize=12, fontweight="bold", va="top",
    )

    # --- Bottom: forest plot panel ---
    ax = fig.add_subplot(gs[1])
    n = len(rows)
    y = np.arange(n)[::-1]

    ax.add_patch(Rectangle(
        (raw_acc, -0.5), 1 - raw_acc, n,
        facecolor="#d8eed4", alpha=0.35, edgecolor="none", zorder=0,
    ))
    ax.add_patch(Rectangle(
        (0, -0.5), raw_acc, n,
        facecolor="#f4d4d4", alpha=0.30, edgecolor="none", zorder=0,
    ))
    ax.axvline(
        raw_acc, color="#444444", linewidth=1.2, linestyle="--", zorder=1,
    )
    ax.axvline(1 / 3, color="#bbbbbb", linewidth=0.7, linestyle=":", zorder=1)

    for i, (tag, label, color, agg, sess) in enumerate(rows):
        se = float(sess.std() / np.sqrt(len(sess))) if len(sess) > 1 else 0.0
        ax.errorbar(
            agg, y[i], xerr=se, fmt="o", color=color, markersize=7,
            ecolor=color, elinewidth=1.0, capsize=3,
            markeredgewidth=0, zorder=3,
        )
        ax.text(
            agg + 0.012, y[i], f"{agg*100:.1f}%",
            va="center", ha="left", fontsize=8, color="#222222", zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in rows], fontsize=8.5)
    ax.set_xlabel(
        "Trial-vote response decoding accuracy", fontsize=8.5,
    )
    ax.set_xlim(0.30, 0.86)
    ax.set_ylim(-0.5, n - 0.5)

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
        1 / 3 + 0.005, -0.45,
        "chance",
        fontsize=7, color="#999999",
        ha="left", va="top",
    )
    ax.text(
        -0.06, n - 0.5, "c",
        transform=ax.transData,
        fontsize=12, fontweight="bold", va="top",
    )
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#888888")

    fig.suptitle(
        "Forecaster architecture determines behavioral decoding accuracy",
        fontsize=11, fontweight="bold", y=0.99,
    )

    out_dir = PROJECT_ROOT / "docs" / "neurips_neurocog" / "figures" / "mockups"
    save_figure(fig, "hero_B_pipeline", out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
