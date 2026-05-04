"""
Figure 2: Evaluation Protocol in Action.

Panels:
  (a) Pareto frontier — params vs weighted Pearson r
  (b) Multi-axis radar — 6 normalized metrics
  (c) Ceiling efficiency bars — % of oracle ceiling

Data source: TABLE1 from data.py (pure Python, no S3 needed).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from figures.style import (
    apply_style, COLORS, MARKERS, MODEL_ORDER,
    add_panel_label, save_figure, TEXT_WIDTH,
)
from figures.data import TABLE1, CEILING_R_EMPIRICAL, CEILING_EFFICIENCY_MEDIAN


def make_pareto(ax):
    """Panel (a): Pareto frontier — params vs weighted r."""

    # Plot each architecture
    for name in MODEL_ORDER:
        d = TABLE1[name]
        ax.scatter(d['params_M'], d['weighted_r'],
                   c=COLORS[name], marker=MARKERS[name],
                   s=90, zorder=3, edgecolors='white', linewidths=0.5)

    # Manual annotations — staggered to prevent the dense top-cluster
    # (Mamba/HGRN2/Transformer at ~2 M params, ~0.49 Wt-r) from
    # piling labels on top of each other.
    offsets = {
        'Mamba':       (0.06,  0.008, 'left'),    # up + right
        'HGRN2':       (-0.08, 0.000, 'right'),   # left
        'Transformer': (0.06, -0.006, 'left'),    # down + right
        'GatedDelta':  (-0.08, 0.000, 'right'),   # left
        'LRU':         (-0.08, 0.000, 'right'),   # left
        'LSTM':        (-0.08, 0.000, 'right'),   # left
        'SNN':         ( 0.08, 0.000, 'left'),    # right
    }
    for name in MODEL_ORDER:
        d = TABLE1[name]
        dx, dy, ha = offsets[name]
        ax.annotate(name, (d['params_M'], d['weighted_r']),
                    xytext=(d['params_M'] + dx, d['weighted_r'] + dy),
                    fontsize=6.5, color=COLORS[name], fontweight='bold',
                    ha=ha, va='center')

    # Axes
    ax.set_xlabel('Parameters (M)')
    ax.set_ylabel('Weighted Pearson $r$')
    ax.set_xlim(0.4, 2.5)
    ax.set_ylim(0.42, 0.56)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
    add_panel_label(ax, 'a')


def make_radar(ax):
    """Panel (b): Multi-axis radar profile for all architectures."""
    # Radar categories (6 axes)
    categories = ['Wt-$r$', 'Pop Rate $r$', 'Spatial $r$',
                  'Cosine', '1 $-$ MAE', 'Param Eff.']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    # Hide the cartesian axes placeholder
    ax.set_visible(False)

    # Create polar subplot inset to prevent label bleed
    fig = ax.get_figure()
    pos = ax.get_position()
    inset = 0.04
    ax_polar = fig.add_axes(
        [pos.x0 + inset, pos.y0 + 0.12,
         pos.width - 2 * inset, pos.height - 0.12],
        polar=True
    )

    # Normalization reference
    max_params = max(TABLE1[m]['params_M'] for m in MODEL_ORDER)
    best_r = max(TABLE1[m]['weighted_r'] for m in MODEL_ORDER)

    def normalize(name):
        """Normalize metrics to [0, 1] for radar display."""
        d = TABLE1[name]
        return [
            d['weighted_r'] / best_r,                       # vs best model
            d['pop_rate_r'],                                # already 0-1
            d['spatial_r'],                                 # already 0-1
            d['cosine_sim'],                                # already 0-1
            1.0 - d['mae'],                                 # invert MAE
            1.0 - (d['params_M'] / (max_params * 1.2)),     # smaller=better
        ]

    # Plot each architecture
    for name in MODEL_ORDER:
        vals = normalize(name)
        vals += vals[:1]
        ax_polar.plot(angles, vals, color=COLORS[name], linewidth=1.5,
                      label=name, zorder=3)
        ax_polar.fill(angles, vals, color=COLORS[name], alpha=0.08)

    # Rotate so the first category (Wt-r) sits at the top, away from
    # adjacent panels (a) on the left and (c) on the right that would
    # otherwise be intersected by the right- and left-edge axis labels.
    ax_polar.set_theta_offset(np.pi / 2)
    ax_polar.set_theta_direction(-1)  # clockwise; reads natural

    # Style the polar axes
    ax_polar.set_xticks(angles[:-1])
    ax_polar.set_xticklabels(categories, fontsize=7)
    ax_polar.set_ylim(0, 1.05)
    ax_polar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_polar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                             fontsize=5.5, color='#999999')
    ax_polar.tick_params(axis='x', pad=8)
    ax_polar.set_rlabel_position(45)
    ax_polar.grid(True, color=COLORS['grid'], linewidth=0.5)
    ax_polar.spines['polar'].set_color(COLORS['grid'])
    ax_polar.spines['polar'].set_linewidth(0.5)

    # Legend below the radar — pushed lower so the bottom Cosine axis
    # label has room above it.
    ax_polar.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
                    ncol=3, fontsize=6, columnspacing=0.8,
                    handlelength=1.2, frameon=False)

    # Panel label — placed top-left of the inset, well outside the radar
    # circle so it doesn't crash into rotated axis labels.
    ax_polar.text(-0.18, 1.18, 'b.', transform=ax_polar.transAxes,
                  fontsize=14, fontweight='bold', fontfamily='sans-serif',
                  color='#222222', va='top', ha='left')


def make_ceiling(ax):
    """Panel (c): Empirical oracle ceiling efficiency bars."""
    # Compute % of empirical oracle ceiling
    # Using oracle_efficiency_median per architecture as the reference
    # For the bar chart, we show relative performance vs best model
    best_r = max(TABLE1[m]['weighted_r'] for m in MODEL_ORDER)
    efficiencies = {name: TABLE1[name]['weighted_r'] / best_r * 100
                    for name in MODEL_ORDER}

    # Sort by efficiency (descending)
    sorted_models = sorted(efficiencies, key=efficiencies.get, reverse=True)
    y_pos = np.arange(len(sorted_models))

    # Draw bars
    bars = ax.barh(y_pos,
                   [efficiencies[m] for m in sorted_models],
                   height=0.55, zorder=3,
                   color=[COLORS[m] for m in sorted_models],
                   edgecolor='white', linewidth=0.5)

    # Percentage labels inside bars
    for bar, name in zip(bars, sorted_models):
        pct = efficiencies[name]
        ax.text(bar.get_width() - 1.5, bar.get_y() + bar.get_height() / 2,
                f'{pct:.0f}%', fontsize=7, fontweight='bold',
                color='white', ha='right', va='center')

    # 100% reference (best model = Mamba)
    ax.axvline(x=100, color=COLORS['Oracle'], linewidth=1.0,
               linestyle='--', zorder=2)

    # Axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_models, fontsize=7.5)
    ax.set_xlabel('% of Best Model')
    ax.set_xlim(0, 110)
    ax.invert_yaxis()
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    add_panel_label(ax, 'c', x=-0.18)


def generate():
    """Generate and save Figure 2."""
    print('Generating Figure 2: Evaluation Protocol...')

    fig = plt.figure(figsize=(TEXT_WIDTH, 3.2))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.3, 0.8],
                           wspace=0.45, left=0.08, right=0.95,
                           top=0.88, bottom=0.18)

    make_pareto(fig.add_subplot(gs[0, 0]))
    make_radar(fig.add_subplot(gs[0, 1]))
    make_ceiling(fig.add_subplot(gs[0, 2]))

    save_figure(fig, 'figure2_evaluation_protocol')
    plt.close(fig)


if __name__ == '__main__':
    apply_style()
    generate()
