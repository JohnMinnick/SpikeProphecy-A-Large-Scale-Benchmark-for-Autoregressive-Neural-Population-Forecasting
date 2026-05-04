"""
Figure 3: Findings Enabled by the Benchmark.

Panels (2026-04-27 layout):
  (a) ANCOVA-adjusted brain-region predictability dot plot
  (b) Mean per-neuron r by Fano factor bin (all 7 architectures)

Population-shuffle panel previously here was moved to the companion
neurocog paper, where its biology framing fits more directly.

Data source: Hardcoded summary stats from kosmos_tier1_analysis +
NRP-recomputed 7-arch Fano-stratified per-neuron r.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from figures.style import (
    apply_style, COLORS, MODEL_ORDER,
    add_panel_label, despine, save_figure, TEXT_WIDTH,
)
from figures.style import MARKERS
from figures.data import (
    REGION_ORDER, REGION_DATA, REGION_DATA_BY_ARCH,
    FANO_BINS, FANO_DATA, SHUFFLE_SUMMARY,
)


def make_region_bars(ax):
    """Panel (a): ANCOVA-adjusted brain-region predictability per
    architecture.  Demonstrates that the predictability hierarchy is
    reproducible across architectures."""
    short_region_labels = {
        'Sensory Cortex': 'Sens',
        'Motor Cortex': 'Motor',
        'Thalamus': 'Thal',
        'Midbrain/\nBrainstem': 'Mid',
        'Basal\nGanglia': 'BG',
        'Frontal/\nAssociation': 'Front',
        'Limbic/\nOther': 'Limb',
        'Hippocampal': 'Hipp',
    }

    # Order: ascending by Mamba's adjusted r (canonical reference);
    # other archs drop on top of the same x positions so the
    # cross-arch hierarchy stability is visible at a glance.
    arch_order = [a for a in ('Mamba', 'HGRN2', 'Transformer',
                              'GatedDelta', 'LRU', 'LSTM', 'SNN')
                  if a in REGION_DATA_BY_ARCH]
    if 'Mamba' in REGION_DATA_BY_ARCH:
        ref_arch = 'Mamba'
    else:
        ref_arch = arch_order[0]
    sorted_regions = sorted(
        REGION_ORDER,
        key=lambda r: REGION_DATA_BY_ARCH[ref_arch][r]['adjusted'],
    )
    x_pos = np.arange(len(sorted_regions))

    # Faint vertical guides per region
    for xi in x_pos:
        ax.axvline(xi, color='#EEEEEE', linewidth=0.5, zorder=0)

    # One marker per (arch, region) at the ANCOVA-adjusted r.
    for arch in arch_order:
        d = REGION_DATA_BY_ARCH[arch]
        adj = [d[r]['adjusted'] for r in sorted_regions]
        ax.plot(x_pos, adj, color=COLORS[arch], marker=MARKERS[arch],
                markersize=5, linewidth=0.8, alpha=0.85,
                markeredgecolor='white', markeredgewidth=0.4,
                label=arch, zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [short_region_labels.get(r, r) for r in sorted_regions],
        rotation=40, ha='right', fontsize=7, rotation_mode='anchor',
    )
    ax.set_ylabel('ANCOVA-adjusted $r$')
    ax.set_ylim(0.0, 0.22)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.3, color='#EEEEEE')
    ax.set_axisbelow(True)
    # Legend above the plot — too many archs for inside.  Keep the
    # panel label well above the legend row to avoid visual overlap.
    ax.legend(fontsize=6, loc='upper center',
              bbox_to_anchor=(0.55, 1.04),
              ncol=len(arch_order), columnspacing=0.6,
              handlelength=1.0, handletextpad=0.3,
              borderaxespad=0.0, frameon=False)
    add_panel_label(ax, 'a', y=1.20)


def make_fano_bars(ax):
    """Panel (b): Mean per-neuron r by Fano factor bin, all archs in
    FANO_DATA (filled in from NRP recomputation)."""
    # Iterate over whatever architectures FANO_DATA carries — keeps
    # the figure in sync as 4-arch -> 7-arch updates land.
    preferred_order = ['Mamba', 'HGRN2', 'Transformer', 'GatedDelta',
                       'LRU', 'LSTM', 'SNN']
    fano_models = [n for n in preferred_order if n in FANO_DATA]
    n_bins = len(FANO_BINS)
    n_models = len(fano_models)
    x = np.arange(n_bins)
    width = 0.78 / max(n_models, 1)

    # Draw grouped bars
    for i, name in enumerate(fano_models):
        vals = FANO_DATA[name]
        ax.bar(x + i * width, vals, width,
               label=name, color=COLORS[name], alpha=0.85,
               edgecolor='white', linewidth=0.25)

    # Sub-Poisson annotation (shaded region for first two bins)
    ax.axvspan(-0.5, 1.5, alpha=0.08, color='#FF6B6B', zorder=0)
    # Place "Sub-Poisson" label above the tallest bar in the shaded
    # region.  With 7 archs, Mamba's FF<0.8 bar reaches ~0.32, so the
    # old y=0.78 (axes coords) overlapped its top — bump to y=0.88
    # (and switch to va='bottom' so the text grows upward, keeping
    # clearance from the bars regardless of small height changes).
    ax.text(0.14, 0.88, 'Sub-Poisson', transform=ax.transAxes,
            fontsize=6, color='#CC3333', ha='center', style='italic',
            va='bottom')

    # Axes
    ax.set_xticks(x + (n_models - 1) * width / 2)
    short_labels = ['<0.8', '0.8-1.0', '1.0-1.2', '1.2-1.5', '>=1.5']
    ax.set_xticklabels(short_labels, fontsize=6.5)
    ax.set_xlabel('Fano Factor Bin')
    ax.set_ylabel('Mean Per-Neuron $r$')
    # Legend along the top (one or two rows depending on arch count) —
    # avoids the Sub-Poisson annotation on the left.
    n_models = len([n for n in ('Mamba','HGRN2','Transformer','GatedDelta',
                                'LRU','LSTM','SNN') if n in FANO_DATA])
    legend_ncol = 4 if n_models <= 4 else 7
    ax.legend(fontsize=6, loc='upper center',
              bbox_to_anchor=(0.55, 1.02),
              ncol=legend_ncol, columnspacing=0.6, handlelength=1.0,
              handletextpad=0.3, borderaxespad=0.0, frameon=False)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.18)
    add_panel_label(ax, 'b', y=1.20)


def make_shuffle_scatter(ax):
    """Panel (c): Original vs Shuffled per-neuron r scatter."""
    # Try to load real scatter data from S3; fall back to synthetic
    try:
        from figures.data import load_shuffle_scatter
        print('  Loading shuffle scatter data from S3...')
        scatter = load_shuffle_scatter(max_sessions=39)
        orig = scatter['original_r']
        shuf = scatter['shuffled_r']
    except Exception as e:
        print(f'  S3 unavailable ({e}), using synthetic scatter...')
        # Generate synthetic data matching known statistics
        rng = np.random.RandomState(42)
        n = SHUFFLE_SUMMARY['n_neurons']
        orig = np.abs(rng.exponential(scale=0.15, size=n))
        slope = SHUFFLE_SUMMARY['trend_slope']
        intercept = SHUFFLE_SUMMARY['trend_intercept']
        shuf = slope * orig + intercept + rng.normal(0, 0.04, n)
        shuf = np.clip(shuf, -0.1, None)

    # Scatter plot
    ax.scatter(orig, shuf, s=0.3, alpha=0.15, color=COLORS['LRU'],
               rasterized=True, zorder=2)

    # Unity line (no drop)
    lim = max(orig.max(), shuf.max()) * 1.05
    ax.plot([0, lim], [0, lim], '--', color='#999999', linewidth=0.8,
            label='No drop ($y=x$)', zorder=3)

    # Trend line
    slope = SHUFFLE_SUMMARY['trend_slope']
    intercept = SHUFFLE_SUMMARY['trend_intercept']
    x_trend = np.linspace(0, 0.9, 100)
    ax.plot(x_trend, slope * x_trend + intercept,
            color=COLORS['Mamba'], linewidth=1.2,
            label=f'Trend: $y={slope}x{intercept:+.3f}$', zorder=4)

    # Mean drop annotation
    ax.text(0.55, 0.15, f'Mean drop: {SHUFFLE_SUMMARY["mean_drop_pct"]}%',
            fontsize=7, fontweight='bold', color=COLORS['Mamba'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['Mamba'], alpha=0.8))

    # Axes
    ax.set_xlabel('Original Pearson $r$')
    ax.set_ylabel('Shuffled Pearson $r$')
    ax.set_xlim(0, 0.95)
    ax.set_ylim(-0.05, 0.95)
    ax.legend(fontsize=5.5, loc='upper left')
    add_panel_label(ax, 'c')


def generate():
    """Generate and save Figure 3."""
    print('Generating Figure 3: Findings...')

    # 2-panel layout (shuffle moved to neurocog paper).
    fig = plt.figure(figsize=(TEXT_WIDTH, 3.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.9, 1.2],
                           wspace=0.40, left=0.08, right=0.96,
                           top=0.86, bottom=0.27)

    make_region_bars(fig.add_subplot(gs[0, 0]))
    make_fano_bars(fig.add_subplot(gs[0, 1]))

    save_figure(fig, 'figure3_findings')
    plt.close(fig)


if __name__ == '__main__':
    apply_style()
    generate()
