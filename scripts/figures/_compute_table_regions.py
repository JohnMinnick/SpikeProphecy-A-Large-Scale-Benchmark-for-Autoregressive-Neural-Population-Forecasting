"""Regenerate Table 3 (tab:regions) values using the canonical
8-system grouping that Figure 2(a) uses, with Mamba per-neuron r
as the reference, plus %sub-Poisson per system.
"""

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from figures._compute_per_region_per_arch import (
    load_canonical_stats, load_region_mapping,
    assign_regions, map_to_systems,
    load_per_neuron_for_arch,
)

fano, sess_idx, neur_idx = load_canonical_stats()
region_lookup = load_region_mapping()
allen = assign_regions(sess_idx, neur_idx, region_lookup)
sys_full = map_to_systems(allen)

mamba = load_per_neuron_for_arch("mamba")
pn_r = mamba["pearson_r"]

# Group by functional system
systems = ["Motor Cortex", "Midbrain/\nBrainstem", "Thalamus",
           "Sensory Cortex", "Frontal/\nAssociation",
           "Hippocampal", "Limbic/\nOther", "Basal\nGanglia"]

display = {
    "Motor Cortex": "Motor cortex",
    "Midbrain/\nBrainstem": "Midbrain/SC",
    "Thalamus": "Thalamus",
    "Sensory Cortex": "Sensory cortex",
    "Frontal/\nAssociation": "Frontal/assoc",
    "Hippocampal": "Hippocampal",
    "Limbic/\nOther": "Limbic/other",
    "Basal\nGanglia": "Basal ganglia",
}

print("Region group        N      Mean r   %sub-P")
results = []
for s in systems:
    mask = (sys_full == s) & ~np.isnan(pn_r) & ~np.isnan(fano)
    n = int(mask.sum())
    mean_r = float(np.mean(pn_r[mask]))
    pct_sub = 100.0 * float(np.mean(fano[mask] < 1.0))
    results.append((s, n, mean_r, pct_sub))

# Sort by mean r descending
results.sort(key=lambda x: -x[2])
total = 0
for s, n, m, p in results:
    print(f"  {display[s]:18s} {n:5d}   {m:.3f}   {p:4.1f}%")
    total += n
print(f"\nTotal: {total} neurons across {len(systems)} systems")

print("\nLaTeX rows:")
for s, n, m, p in results:
    n_str = f"{n:,}".replace(",", "{,}")
    print(f"    {display[s]:14s} & {n_str} & {m:.3f} & {p:.0f}\\% \\\\")
