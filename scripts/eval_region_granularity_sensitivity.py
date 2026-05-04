"""Brain-region grouping sensitivity (reviewer Q2).

Reviewer asked: "How sensitive is R^2 = 0.28 to the choice of grouping (8
Allen functional systems)? At finer/coarser granularity does it change?"

This script re-runs the ANCOVA model_r ~ log_rate + fano + region at three
granularities:

  Fine    -- top-K raw Allen acronyms (drop sparse-cell regions for stability)
  Medium  -- 8 functional systems (current paper)
  Coarse  -- 4 broad classes (Cortex / Subcortex / Hippocampal / Other)

For each granularity, report R^2, p-value of the F-test on region indicators
(after partialling out covariates), and the number of regions retained.

Inputs (all already on disk):
  outputs/eval_analysis/per_neuron_stats.json    -- mean_rate, fano, model_r
  outputs/eval_analysis/brain_region_mapping.json -- per-session neuron_regions
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PER_NEURON = ROOT / "outputs/eval_analysis/per_neuron_stats.json"
REGION_MAP = ROOT / "outputs/eval_analysis/brain_region_mapping.json"
OUT = ROOT / "outputs/eval_analysis/region_granularity_sensitivity.json"

# 8 functional systems used in the paper
FUNCTIONAL_8 = {
    # Sensory cortex
    "VISp": "Sensory Cortex", "VISam": "Sensory Cortex",
    "VISpm": "Sensory Cortex", "VISl": "Sensory Cortex",
    "VISal": "Sensory Cortex", "VISli": "Sensory Cortex",
    "VISa": "Sensory Cortex", "VISrl": "Sensory Cortex",
    "VISpor": "Sensory Cortex", "AUDp": "Sensory Cortex",
    "AUDpo": "Sensory Cortex", "SSp": "Sensory Cortex",
    "SSs": "Sensory Cortex", "PIR": "Sensory Cortex",
    # Motor cortex
    "MOp": "Motor Cortex", "MOs": "Motor Cortex",
    # Frontal/Association
    "ACA": "Frontal/Association", "PL": "Frontal/Association",
    "ILA": "Frontal/Association", "ORB": "Frontal/Association",
    "ORBm": "Frontal/Association", "ORBl": "Frontal/Association",
    "FRP": "Frontal/Association", "AI": "Frontal/Association",
    "AId": "Frontal/Association", "AIv": "Frontal/Association",
    "AIp": "Frontal/Association",
    "RSP": "Frontal/Association", "RSPagl": "Frontal/Association",
    "RSPv": "Frontal/Association", "RSPd": "Frontal/Association",
    "PTLp": "Frontal/Association", "TEa": "Frontal/Association",
    "ECT": "Frontal/Association", "PERI": "Frontal/Association",
    # Hippocampal
    "CA1": "Hippocampal", "CA2": "Hippocampal", "CA3": "Hippocampal",
    "DG": "Hippocampal", "SUB": "Hippocampal", "POST": "Hippocampal",
    "PRE": "Hippocampal", "ProS": "Hippocampal",
    # Thalamus
    "TH": "Thalamus", "VPM": "Thalamus", "VPL": "Thalamus",
    "LGd": "Thalamus", "LP": "Thalamus", "PO": "Thalamus",
    "MD": "Thalamus", "RT": "Thalamus", "PVT": "Thalamus",
    "AV": "Thalamus", "AM": "Thalamus", "AD": "Thalamus",
    "VAL": "Thalamus", "VM": "Thalamus", "CL": "Thalamus",
    "CM": "Thalamus", "PCN": "Thalamus", "MG": "Thalamus",
    "RH": "Thalamus", "PF": "Thalamus", "IAD": "Thalamus",
    "LD": "Thalamus", "LH": "Thalamus", "POL": "Thalamus",
    "SGN": "Thalamus", "SPF": "Thalamus", "Eth": "Thalamus",
    # Midbrain / Brainstem
    "MRN": "Midbrain/Brainstem", "SCm": "Midbrain/Brainstem",
    "SCs": "Midbrain/Brainstem", "SCdg": "Midbrain/Brainstem",
    "SCig": "Midbrain/Brainstem", "SCop": "Midbrain/Brainstem",
    "SCsg": "Midbrain/Brainstem", "SCzo": "Midbrain/Brainstem",
    "PAG": "Midbrain/Brainstem", "APN": "Midbrain/Brainstem",
    "MB": "Midbrain/Brainstem", "RN": "Midbrain/Brainstem",
    "VTA": "Midbrain/Brainstem", "SN": "Midbrain/Brainstem",
    "SNr": "Midbrain/Brainstem", "SNc": "Midbrain/Brainstem",
    "PPN": "Midbrain/Brainstem", "NB": "Midbrain/Brainstem",
    "ZI": "Midbrain/Brainstem",
    # Basal Ganglia
    "STR": "Basal Ganglia", "CP": "Basal Ganglia",
    "ACB": "Basal Ganglia", "OT": "Basal Ganglia",
    "GPe": "Basal Ganglia", "GPi": "Basal Ganglia",
    "STN": "Basal Ganglia", "FS": "Basal Ganglia",
    # Limbic / Other
    "LS": "Limbic/Other", "LSr": "Limbic/Other", "LSc": "Limbic/Other",
    "BLA": "Limbic/Other", "MEA": "Limbic/Other", "BST": "Limbic/Other",
    "EP": "Limbic/Other", "EPd": "Limbic/Other",
    "MS": "Limbic/Other", "TRS": "Limbic/Other",
}

# 4 coarse classes
COARSE_4 = {
    "Sensory Cortex": "Cortex", "Motor Cortex": "Cortex",
    "Frontal/Association": "Cortex",
    "Hippocampal": "Hippocampal",
    "Thalamus": "Subcortex", "Midbrain/Brainstem": "Subcortex",
    "Basal Ganglia": "Subcortex",
    "Limbic/Other": "Other",
}


def fit_ancova(model_r, log_rate, fano, region_idx, n_regions):
    """OLS fit with covariates and dummy region variables.

    Returns dict with R^2, R^2 of region indicators only (incremental),
    n_neurons, n_regions, F-statistic for region.
    """
    n = len(model_r)
    n_dummies = n_regions - 1  # dropped reference category
    # Full model: [intercept, log_rate, fano, region_dummies]
    X_full = np.ones((n, 3 + n_dummies))
    X_full[:, 1] = log_rate
    X_full[:, 2] = fano
    for i in range(n_dummies):
        X_full[:, 3 + i] = (region_idx == i).astype(float)
    beta, *_ = np.linalg.lstsq(X_full, model_r, rcond=None)
    y_pred = X_full @ beta
    ss_res_full = float(np.sum((model_r - y_pred) ** 2))
    ss_tot = float(np.sum((model_r - model_r.mean()) ** 2))
    r2_full = 1 - ss_res_full / ss_tot

    # Reduced model: covariates only (no region dummies)
    X_red = np.ones((n, 3))
    X_red[:, 1] = log_rate
    X_red[:, 2] = fano
    beta_r, *_ = np.linalg.lstsq(X_red, model_r, rcond=None)
    y_pred_r = X_red @ beta_r
    ss_res_red = float(np.sum((model_r - y_pred_r) ** 2))
    r2_red = 1 - ss_res_red / ss_tot

    # Incremental R^2 from adding region dummies
    r2_increment = r2_full - r2_red

    # Partial F-test: F = ((SS_red - SS_full) / q) / (SS_full / (n - p))
    p_full = 3 + n_dummies
    q = n_dummies
    if q > 0 and (n - p_full) > 0:
        f_stat = ((ss_res_red - ss_res_full) / q) / (ss_res_full / (n - p_full))
    else:
        f_stat = float("nan")

    return {
        "n_neurons": int(n),
        "n_regions": int(n_regions),
        "r2_covariates_only": round(r2_red, 4),
        "r2_with_regions": round(r2_full, 4),
        "r2_region_increment": round(r2_increment, 4),
        "f_stat_region": round(f_stat, 2),
    }


def main():
    print("Loading per-neuron stats...")
    pn = json.load(open(PER_NEURON))
    rm = json.load(open(REGION_MAP))
    print(f"  per_neuron_stats: {len(pn)} entries")
    print(f"  brain_region_mapping: {rm['n_sessions']} sessions, "
          f"{rm['n_regions']} unique regions")

    # Join: for each per_neuron entry, look up its actual region
    # using (session, neuron_idx) -> sessions[session].neuron_regions[neuron_idx]
    rows = []
    sessions = rm["sessions"]
    n_skipped = 0
    for entry in pn:
        sidx = str(entry["session"])
        nidx = entry["neuron"]
        if sidx not in sessions:
            n_skipped += 1
            continue
        nr = sessions[sidx].get("neuron_regions")
        if nr is None or nidx >= len(nr):
            n_skipped += 1
            continue
        region = nr[nidx]
        rows.append({
            "model_r": float(entry["model_r"]),
            "log_rate": float(np.log(max(entry["mean_rate"], 1e-6))),
            "fano": float(entry["fano_factor"]),
            "region": region,
        })
    print(f"  joined neurons: {len(rows)} (skipped {n_skipped})")

    model_r = np.array([r["model_r"] for r in rows])
    log_rate = np.array([r["log_rate"] for r in rows])
    fano = np.array([r["fano"] for r in rows])
    raw_region = np.array([r["region"] for r in rows])

    results = {"granularities": {}}

    # --- FINE: top-K raw Allen regions (filter by min count for stability) ---
    MIN_COUNT_FINE = 100  # require at least 100 neurons for a region
    counts = {}
    for r in raw_region:
        counts[r] = counts.get(r, 0) + 1
    keep_fine = {r for r, c in counts.items() if c >= MIN_COUNT_FINE}
    mask_fine = np.array([r in keep_fine for r in raw_region])
    fine_regions = sorted(keep_fine)
    fine_idx_map = {r: i for i, r in enumerate(fine_regions)}
    fine_idx = np.array([fine_idx_map[r] for r in raw_region[mask_fine]])
    print(f"\nFINE: {len(fine_regions)} raw Allen regions "
          f"(min {MIN_COUNT_FINE} neurons each), "
          f"{mask_fine.sum()} neurons retained")
    results["granularities"]["fine_raw_allen"] = fit_ancova(
        model_r[mask_fine], log_rate[mask_fine], fano[mask_fine],
        fine_idx, len(fine_regions),
    )
    results["granularities"]["fine_raw_allen"]["min_neurons_per_region"] = MIN_COUNT_FINE
    results["granularities"]["fine_raw_allen"]["regions"] = fine_regions

    # --- MEDIUM: 8 functional systems ---
    fs_label = np.array([FUNCTIONAL_8.get(r, "Unclassified") for r in raw_region])
    keep_med = fs_label != "Unclassified"
    med_labels = sorted(set(fs_label[keep_med]))
    med_idx_map = {s: i for i, s in enumerate(med_labels)}
    med_idx = np.array([med_idx_map[s] for s in fs_label[keep_med]])
    print(f"MEDIUM: {len(med_labels)} functional systems, "
          f"{keep_med.sum()} neurons retained")
    results["granularities"]["medium_8_systems"] = fit_ancova(
        model_r[keep_med], log_rate[keep_med], fano[keep_med],
        med_idx, len(med_labels),
    )
    results["granularities"]["medium_8_systems"]["regions"] = med_labels

    # --- COARSE: 4 broad classes ---
    coarse_label = np.array([
        COARSE_4.get(FUNCTIONAL_8.get(r, ""), "Unclassified") for r in raw_region
    ])
    keep_coarse = coarse_label != "Unclassified"
    coarse_labels = sorted(set(coarse_label[keep_coarse]))
    coarse_idx_map = {s: i for i, s in enumerate(coarse_labels)}
    coarse_idx = np.array([coarse_idx_map[s] for s in coarse_label[keep_coarse]])
    print(f"COARSE: {len(coarse_labels)} broad classes, "
          f"{keep_coarse.sum()} neurons retained")
    results["granularities"]["coarse_4_classes"] = fit_ancova(
        model_r[keep_coarse], log_rate[keep_coarse], fano[keep_coarse],
        coarse_idx, len(coarse_labels),
    )
    results["granularities"]["coarse_4_classes"]["regions"] = coarse_labels

    # --- Summary ---
    print()
    print("Granularity sensitivity (ANCOVA):")
    print(f"  {'Granularity':<24} {'k':>4} {'N':>7} {'R^2_cov':>9} {'R^2_full':>9} {'incr R^2':>9}")
    for label, r in results["granularities"].items():
        print(f"  {label:<24} {r['n_regions']:>4} {r['n_neurons']:>7} "
              f"{r['r2_covariates_only']:>9.4f} {r['r2_with_regions']:>9.4f} "
              f"{r['r2_region_increment']:>9.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
