"""Compute session-4 Fano-stratified per-neuron r for all 7 architectures.

Stratifies session-4 neurons into 5 Fano-factor bins (matching the
existing FANO_BINS schema in figures.data) and computes the mean
per-neuron Pearson r within each bin for each architecture.

This gives us a 7-arch panel for Figure 2(b) at the cost of being
session-specific instead of 39-session-aggregate.
"""

from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
import json

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data" / "figure_cache" / "multi_arch_session4.npz"
OUT = ROOT / "data" / "figure_cache" / "fano_session4_7arch.json"


def main():
    d = np.load(str(NPZ))
    gt = d["gt"]  # (T, N)
    T, N = gt.shape

    # Per-neuron Fano factor (only meaningful where neuron fires)
    means = gt.mean(axis=0)
    vars_ = gt.var(axis=0)
    fano = np.full(N, np.nan)
    valid = means > 1e-3
    fano[valid] = vars_[valid] / means[valid]
    print(f"Computed Fano for {valid.sum()} of {N} neurons")
    print(f"Fano dist: median={np.nanmedian(fano):.3f}, "
          f"mean={np.nanmean(fano):.3f}")

    # Fano bins (matching existing schema)
    bin_edges = [0.0, 0.8, 1.0, 1.2, 1.5, np.inf]
    bin_labels = ["FF<0.8", "0.8<=FF<1.0", "1.0<=FF<1.2",
                  "1.2<=FF<1.5", "FF>=1.5"]
    bin_idx = np.digitize(fano, bin_edges) - 1  # 0..4 for valid

    print("\nNeuron count per Fano bin:")
    for i, lab in enumerate(bin_labels):
        n_in_bin = ((bin_idx == i) & valid).sum()
        print(f"  {lab}: {n_in_bin}")

    # Per-arch per-bin per-neuron r
    archs = [
        ("mamba_rates",       "Mamba"),
        ("hgrn2_rates",       "HGRN2"),
        ("transformer_rates", "Transformer"),
        ("gated_delta_rates", "GatedDelta"),
        ("lru_rates",         "LRU"),
        ("lstm_rates",        "LSTM"),
        ("snn_rates",         "SNN"),
    ]

    fano_data = {}
    for key, name in archs:
        if key not in d.files:
            print(f"  WARN: {key} not in NPZ")
            continue
        rates = d[key]
        # Per-neuron r
        pn_r = np.full(N, np.nan)
        for j in range(N):
            if gt[:, j].std() > 0 and rates[:, j].std() > 0:
                pn_r[j] = pearsonr(gt[:, j], rates[:, j])[0]
        # Average per Fano bin
        per_bin = []
        for i in range(len(bin_labels)):
            mask = (bin_idx == i) & valid & ~np.isnan(pn_r)
            if mask.sum() > 0:
                per_bin.append(float(np.mean(pn_r[mask])))
            else:
                per_bin.append(0.0)
        fano_data[name] = per_bin
        print(f"\n  {name:14s}: {[f'{v:.3f}' for v in per_bin]}")

    # Save to JSON for the figure script
    output = {
        "session_idx": 4,
        "n_neurons": int(N),
        "n_active": int(valid.sum()),
        "bin_labels": bin_labels,
        "fano_data": fano_data,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
