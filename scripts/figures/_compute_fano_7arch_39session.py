"""Compute 39-session Fano-stratified per-neuron r for all 7 archs.

Joins:
  - eval-suite/{mamba,transformer,lru,snn}/per_neuron_arrays.npz
    (per-neuron r computed on val split, full 39 sessions)
  - data/figure_cache/per_neuron_data_3arch.npz
    (NRP val-only per-neuron r for HGRN2/GatedDelta/LSTM, plus the
     canonical val-only fano_per_session arrays — same orderings)

Result: FANO_DATA-format dict, all 7 archs at 39-session aggregate.
"""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL_SUITE = ROOT / "outputs" / "eval-suite"
NRP_NPZ = ROOT / "data" / "figure_cache" / "per_neuron_data_3arch.npz"

BIN_EDGES = [0.0, 0.8, 1.0, 1.2, 1.5, np.inf]
BIN_LABELS = ["FF<0.8", "0.8<=FF<1.0", "1.0<=FF<1.2",
              "1.2<=FF<1.5", "FF>=1.5"]


def load_eval_suite_pn_r(arch_lc):
    p = EVAL_SUITE / arch_lc / "per_neuron_arrays.npz"
    d = np.load(str(p), allow_pickle=True)
    return np.array(d["pearson_r"], dtype=np.float64)


def main():
    nrp = np.load(str(NRP_NPZ), allow_pickle=True)
    fano_per_session = nrp["fano_per_session"]
    flat_fano = np.concatenate(
        [np.asarray(f, dtype=np.float64) for f in fano_per_session]
    )
    print(f"Total neurons: {len(flat_fano)}")
    valid = ~np.isnan(flat_fano) & (flat_fano > 0)
    print(f"Valid Fano: {valid.sum()}")

    bin_idx = np.digitize(flat_fano, BIN_EDGES) - 1
    print("Neurons per bin:")
    for i, lbl in enumerate(BIN_LABELS):
        n = ((bin_idx == i) & valid).sum()
        print(f"  {lbl}: {n}")

    fano_data = {}

    # 4 archs from eval-suite
    eval_suite_archs = {
        "Mamba": "mamba",
        "Transformer": "transformer",
        "LRU": "lru",
        "SNN": "snn",
    }
    for display, lc in eval_suite_archs.items():
        pn_r = load_eval_suite_pn_r(lc)
        if len(pn_r) != len(flat_fano):
            print(f"  WARN {display}: {len(pn_r)} vs {len(flat_fano)}")
            continue
        per_bin = []
        for i in range(len(BIN_LABELS)):
            mask = (bin_idx == i) & valid & ~np.isnan(pn_r)
            if mask.sum() > 0:
                per_bin.append(round(float(np.mean(pn_r[mask])), 3))
            else:
                per_bin.append(0.0)
        fano_data[display] = per_bin

    # 3 archs from NRP NPZ
    for arch_disp in ["HGRN2", "GatedDelta", "LSTM"]:
        if f"pn_r__{arch_disp}" not in nrp.files:
            print(f"  WARN: {arch_disp} not in NPZ")
            continue
        pn_per_session = nrp[f"pn_r__{arch_disp}"]
        flat_pn = np.concatenate(
            [np.asarray(p, dtype=np.float64) for p in pn_per_session]
        )
        if len(flat_pn) != len(flat_fano):
            print(f"  WARN {arch_disp}: {len(flat_pn)} vs {len(flat_fano)}")
            continue
        per_bin = []
        for i in range(len(BIN_LABELS)):
            mask = (bin_idx == i) & valid & ~np.isnan(flat_pn)
            if mask.sum() > 0:
                per_bin.append(round(float(np.mean(flat_pn[mask])), 3))
            else:
                per_bin.append(0.0)
        fano_data[arch_disp] = per_bin

    print("\nFANO_DATA = {")
    print("    #              FF<0.8   0.8-1.0  1.0-1.2  1.2-1.5  >=1.5")
    order = ["Mamba", "HGRN2", "Transformer", "GatedDelta",
             "LRU", "LSTM", "SNN"]
    for arch in order:
        if arch not in fano_data:
            continue
        vals = fano_data[arch]
        v_str = "  ".join(f"{v:.3f}" for v in vals)
        print(f"    '{arch}':{' ' * (12 - len(arch))}[{v_str}],")
    print("}")


if __name__ == "__main__":
    main()
