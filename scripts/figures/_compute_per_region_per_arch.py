"""Compute per-arch ANCOVA-adjusted per-region means for Figure 2(a).

For each architecture with a per-neuron-arrays NPZ available, fit
the ANCOVA model
    pearson_r ~ log(mean_rate + 1) + fano_factor + C(region)
and report the marginal per-region mean (covariate-adjusted).
This is the same covariate set used in the canonical
kosmos_tier1_analysis.py (Mamba reference).

Output (printed):
  REGION_DATA_BY_ARCH = {
      'Mamba':       {region: {'raw': float, 'adjusted': float, 'n': int}, ...},
      'LRU':         { ... },
      ...
  }
"""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL_SUITE = ROOT / "outputs" / "eval-suite"
STATS = ROOT / "outputs" / "eval_analysis" / "per_neuron_stats.json"

# Allen-system grouping (matches data.py REGION_ORDER)
FUNCTIONAL_SYSTEMS = {
    "Sensory Cortex": ["VISp", "VISpm", "VISa", "VISl", "VISrl", "VISam",
                        "VISpl", "AUDp", "AUDd", "AUDv", "AUDpo",
                        "SSp-bfd", "SSp-ll", "SSp-m", "SSp-n", "SSp-tr",
                        "SSp-ul", "SSp-un", "SSs", "RSPagl", "RSPd",
                        "RSPv"],
    "Motor Cortex": ["MOs", "MOp"],
    "Thalamus": ["LGd", "LP", "PO", "PoT", "MD", "MG", "PVT", "VPL",
                  "VPM", "VPLpc", "VPMpc", "VAL", "VM", "AV", "AM",
                  "AD", "LD", "LH", "RT", "TH", "IntG"],
    "Midbrain/\nBrainstem": ["SCs", "SCm", "SCop", "SCdg", "SCiw",
                              "SCig", "SCdw", "SCsg", "SCzo", "MRN",
                              "PRNr", "PRNc", "MB", "RPF", "APN", "NPC",
                              "NB", "PAG", "GRN", "VTA", "PRN", "ICe",
                              "MRP", "RR"],
    "Basal\nGanglia": ["CP", "ACB", "GPe", "GPi", "STN", "SNr", "SI",
                        "BST", "MEA", "OT"],
    "Frontal/\nAssociation": ["ACA", "ACAd", "ACAv", "PL", "ILA",
                               "ORB", "ORBl", "ORBm", "ORBvl", "FRP"],
    "Limbic/\nOther": ["PIR", "BMA", "LA", "BLA", "EP", "EPv", "EPd",
                        "TR", "PAA", "AON"],
    "Hippocampal": ["CA1", "CA2", "CA3", "DG", "DG-mo", "DG-po",
                     "DG-sg", "POST", "PRE", "SUB", "ProS", "PAR",
                     "POR", "ENT", "ENTl", "ENTm", "LS", "LSr"],
}

REGION_TO_SYSTEM = {}
for system, regions in FUNCTIONAL_SYSTEMS.items():
    for region in regions:
        REGION_TO_SYSTEM[region] = system


def load_per_neuron_for_arch(name):
    """Return (pearson_r, mean_rate) arrays for an arch.  Region info
    is loaded separately from brain_region_mapping.json since the
    npz's `regions` field is actually session IDs, not Allen acronyms."""
    p = EVAL_SUITE / name / "per_neuron_arrays.npz"
    if not p.exists():
        return None
    d = np.load(str(p), allow_pickle=True)
    return {
        "pearson_r": np.array(d["pearson_r"], dtype=np.float64),
        "mean_rate": np.array(d["mean_rate"], dtype=np.float64),
    }


def load_canonical_stats():
    """Per-neuron metadata (session, neuron, fano, region from Allen)."""
    with open(STATS) as f:
        per_neuron = json.load(f)
    fano = np.array([e.get("fano_factor", np.nan) for e in per_neuron],
                    dtype=np.float64)
    sess_idx = np.array([e["session"] for e in per_neuron], dtype=np.int32)
    neur_idx = np.array([e["neuron"] for e in per_neuron], dtype=np.int32)
    return fano, sess_idx, neur_idx


def load_region_mapping():
    """Build a per-(session_id, neuron_idx) -> Allen acronym lookup."""
    region_map_path = (ROOT / "outputs" / "eval_analysis"
                       / "brain_region_mapping.json")
    with open(region_map_path) as f:
        rm = json.load(f)
    lookup = {}
    for sess_id, sess_data in rm["sessions"].items():
        for ni, region in enumerate(sess_data["neuron_regions"]):
            lookup[(int(sess_id), ni)] = region
    return lookup


def assign_regions(sess_idx, neur_idx, lookup):
    """For each neuron, assign Allen region acronym."""
    return np.array([lookup.get((int(s), int(n)), "unknown")
                     for s, n in zip(sess_idx, neur_idx)], dtype=object)


def map_to_systems(regions):
    """Map the per-neuron Allen acronyms to the 8 functional systems."""
    out = []
    for r in regions:
        # regions might be 'root' or unknown — bucket as None
        if r in REGION_TO_SYSTEM:
            out.append(REGION_TO_SYSTEM[r])
        else:
            out.append(None)
    return np.array(out, dtype=object)


def ancova_adjusted_means(pearson_r, log_rate, fano, system_labels):
    """Fit y = a + b1*log_rate + b2*fano + sum(c_s * I[system=s]) and
    return per-system covariate-adjusted means (LSMeans).

    Uses the standard ANCOVA approach: marginal means are evaluated at
    the grand-mean covariate values."""
    valid = (~np.isnan(pearson_r) & ~np.isnan(log_rate) & ~np.isnan(fano)
             & np.array([s is not None for s in system_labels]))
    y = pearson_r[valid]
    lr = log_rate[valid]
    ff = fano[valid]
    sys = system_labels[valid]
    sys_unique = sorted(set(sys))

    # Build design matrix: intercept, log_rate, fano, dummy per system
    # (drop first as reference)
    n = len(y)
    n_sys = len(sys_unique)
    X = np.zeros((n, 3 + n_sys - 1), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1] = lr
    X[:, 2] = ff
    for i, s in enumerate(sys_unique[1:], start=3):
        X[:, i] = (sys == s).astype(np.float64)

    # OLS
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, b_lr, b_ff = beta[0], beta[1], beta[2]
    sys_effects = np.concatenate([[0.0], beta[3:]])  # reference is 0

    # Adjusted mean per system: intercept + b_lr*mean(lr) + b_ff*mean(ff) + sys_effect
    grand_lr = lr.mean()
    grand_ff = ff.mean()
    adjusted = {}
    raw = {}
    counts = {}
    for s, eff in zip(sys_unique, sys_effects):
        m = sys == s
        raw[s] = float(y[m].mean())
        counts[s] = int(m.sum())
        adjusted[s] = float(intercept + b_lr * grand_lr + b_ff * grand_ff
                             + eff)
    return raw, adjusted, counts


def main():
    print("Loading canonical stats + region mapping...")
    fano_full, sess_idx, neur_idx = load_canonical_stats()
    region_lookup = load_region_mapping()
    allen_regions = assign_regions(sess_idx, neur_idx, region_lookup)
    print(f"  {len(fano_full)} neurons, "
          f"{np.sum(allen_regions != 'unknown')} with Allen labels")
    sys_full = map_to_systems(allen_regions)
    print(f"  {np.sum([s is not None for s in sys_full])} mapped to a "
          f"functional system")

    out = {}
    # mean_rate is the same regardless of arch (it's a property of the
    # ground-truth data) — borrow it from Mamba's eval-suite arrays for
    # the ANCOVA covariate.
    mamba_arrays = load_per_neuron_for_arch("mamba")
    canonical_log_rate = np.log(mamba_arrays["mean_rate"] + 1.0)

    for name in ["mamba", "lru", "transformer", "snn"]:
        d = load_per_neuron_for_arch(name)
        if d is None:
            print(f"  {name}: no data, skip")
            continue
        if len(d["pearson_r"]) != len(fano_full):
            print(f"  {name}: length mismatch ({len(d['pearson_r'])} vs "
                  f"{len(fano_full)}), skip")
            continue
        log_rate = np.log(d["mean_rate"] + 1.0)
        raw, adjusted, counts = ancova_adjusted_means(
            d["pearson_r"], log_rate, fano_full, sys_full
        )
        # Use the display name (capitalize)
        display = {"mamba": "Mamba", "lru": "LRU",
                   "transformer": "Transformer", "snn": "SNN"}[name]
        out[display] = {s: {"raw": raw[s], "adjusted": adjusted[s],
                             "n": counts[s]}
                        for s in raw}
        print(f"  {display}: {sum(counts.values())} neurons, "
              f"{len(raw)} regions")

    # ---- Plug in the 3 new archs from the NRP per_neuron_data.npz ----
    nrp_npz = ROOT / "data" / "figure_cache" / "per_neuron_data_3arch.npz"
    if nrp_npz.exists():
        print(f"\nLoading 3-arch NRP results from {nrp_npz.name}")
        nd = np.load(str(nrp_npz), allow_pickle=True)
        new_arch_names = list(nd["arch_names"])
        # Concatenate per-session arrays in session order to match the
        # canonical (session, neuron) ordering used by per_neuron_stats.
        for arch_disp in new_arch_names:
            pn_per_session = nd[f"pn_r__{arch_disp}"]
            flat_pn = np.concatenate([np.asarray(p, dtype=np.float64)
                                      for p in pn_per_session])
            if len(flat_pn) != len(fano_full):
                print(f"  {arch_disp}: length mismatch "
                      f"({len(flat_pn)} vs {len(fano_full)}), skip")
                continue
            raw, adjusted, counts = ancova_adjusted_means(
                flat_pn, canonical_log_rate, fano_full, sys_full
            )
            out[arch_disp] = {s: {"raw": raw[s], "adjusted": adjusted[s],
                                   "n": counts[s]}
                              for s in raw}
            print(f"  {arch_disp}: {sum(counts.values())} neurons, "
                  f"{len(raw)} regions")

    print("\nREGION_DATA_BY_ARCH = {")
    for arch, data in out.items():
        print(f"    '{arch}': {{")
        for region, vals in data.items():
            r_print = repr(region).replace("'", '"')
            print(f"        {r_print}: {{'n': {vals['n']:5d}, "
                  f"'raw': {vals['raw']:.3f}, "
                  f"'adjusted': {vals['adjusted']:.3f}}},")
        print("    },")
    print("}")

    # Save for downstream
    cache = ROOT / "data" / "figure_cache" / "region_per_arch.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {cache}")


if __name__ == "__main__":
    main()
