"""Region-hierarchy robustness: residualize per-neuron r against log mean firing rate.

Reviewer concern (NeurIPS rev #2): regions with higher mean firing rates or more
stereotyped task-locked responses will be easier to predict for any
architecture, so the apparent ``intrinsic region-level dynamics'' hierarchy
might just track firing-rate differences across regions. Test by:

  1. For each neuron i, compute log10(mean_FR_i) across the full session.
  2. Fit a global linear model: r_i ~ beta * log10(FR_i) + alpha.
  3. Residual r_i^res = r_i - (alpha + beta * log10(FR_i)).
  4. Re-rank regions by mean r^res, compare to original ranking via
     Spearman rho. If the hierarchy survives residualization, it is not
     reducible to firing-rate differences across regions.

Usage:
    python scripts/eval_region_hierarchy_residualized.py
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kruskal, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_JSONS = {
    "1L SNN": PROJECT_ROOT / "outputs/eval_local/multihead_1l_v3_full.json",
    "2L SNN": PROJECT_ROOT / "outputs/eval_local/multihead_2l_v3_full.json",
}
REGION_MAP_PATH = (
    PROJECT_ROOT / "outputs/eval_analysis/brain_region_mapping.json"
)
CACHE_DIR = (
    PROJECT_ROOT / "data/processed/combined_steinmetz_ibl_cache"
)
OUT_PATH = (
    PROJECT_ROOT / "outputs/eval_local/region_hierarchy_residualized.json"
)
MIN_NEURONS_PER_REGION = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("region_residualized")


def load_session_neuron_records(eval_path: Path, region_map: dict) -> list:
    """Return list of (region, r, log_mean_fr) per neuron, Steinmetz only."""
    d = json.load(open(eval_path))
    sessions_map = region_map["sessions"]
    records = []
    for s in d["per_session"]:
        if s.get("source") != "steinmetz":
            continue
        sidx = s["session_idx"]
        if str(sidx) not in sessions_map:
            continue
        regions = sessions_map[str(sidx)].get("neuron_regions")
        r_list = s.get("per_neuron_r")
        if regions is None or r_list is None:
            continue
        # Load the counts to get per-neuron mean firing rate
        counts_path = CACHE_DIR / f"session_{sidx:03d}.npy"
        if not counts_path.exists():
            log.warning("counts missing for session %d", sidx)
            continue
        counts = np.load(counts_path)  # (M, T) int
        # Per-neuron mean spikes per bin (50ms bin -> can convert to Hz, but
        # log of the mean is what we need; constant offset doesn't matter).
        mean_spk_per_bin = counts.mean(axis=1)
        n = min(len(regions), len(r_list), len(mean_spk_per_bin))
        for i in range(n):
            mr = float(mean_spk_per_bin[i])
            if mr <= 0:
                continue  # silent neuron, log undefined
            records.append((regions[i], float(r_list[i]), float(np.log10(mr))))
    return records


def residualize(records):
    """Linear regression r ~ log10(FR), return residuals dict[(region, r_res)]."""
    arr = np.array(records, dtype=object)
    rs = np.array([rec[1] for rec in records], dtype=np.float64)
    log_fr = np.array([rec[2] for rec in records], dtype=np.float64)
    # OLS
    X = np.column_stack([np.ones_like(log_fr), log_fr])
    beta, *_ = np.linalg.lstsq(X, rs, rcond=None)
    intercept, slope = beta[0], beta[1]
    fitted = intercept + slope * log_fr
    residuals = rs - fitted
    log.info(
        "  OLS:  r = %.4f + %.4f * log10(FR), corr(r, logFR)=%.3f",
        intercept, slope, float(np.corrcoef(rs, log_fr)[0, 1]),
    )
    return [(rec[0], float(res), rec[1]) for rec, res in zip(records, residuals)]


def per_region_arrays(records, value_idx: int):
    out = defaultdict(list)
    for rec in records:
        out[rec[0]].append(rec[value_idx])
    return {rg: np.array(rs) for rg, rs in out.items()}


def ranking(arrays, min_n=MIN_NEURONS_PER_REGION):
    means = {rg: float(np.mean(v)) for rg, v in arrays.items() if len(v) >= min_n}
    return sorted(means.keys(), key=lambda r: -means[r]), means


def main():
    region_map = json.load(open(REGION_MAP_PATH))
    results = {"models": {}, "min_neurons_per_region": MIN_NEURONS_PER_REGION}

    for name, path in EVAL_JSONS.items():
        log.info("=" * 60)
        log.info("Model: %s", name)
        records = load_session_neuron_records(path, region_map)
        log.info("  loaded %d (region, r, logFR) records", len(records))
        residualized = residualize(records)

        # Original ranking by raw r
        arrs_raw = per_region_arrays(
            [(r, raw_r, raw_r) for r, _, raw_r in residualized], value_idx=1,
        )
        ranking_raw, means_raw = ranking(arrs_raw)
        # Residualized ranking by r_res
        arrs_res = per_region_arrays(residualized, value_idx=1)
        ranking_res, means_res = ranking(arrs_res)

        common = [r for r in ranking_raw if r in means_res]
        rank_raw = {r: i for i, r in enumerate(ranking_raw)}
        rank_res = {r: i for i, r in enumerate(ranking_res)}
        x = [rank_raw[r] for r in common]
        y = [rank_res[r] for r in common]
        rho_raw_vs_res, _ = spearmanr(x, y)

        # Top-N sanity: how many of the top-10 stay in the top-10?
        top10_raw = set(ranking_raw[:10])
        top10_res = set(ranking_res[:10])
        overlap_top10 = len(top10_raw & top10_res)

        # K-W on residualized values
        groups_res = [
            v for v in arrs_res.values() if len(v) >= MIN_NEURONS_PER_REGION
        ]
        H_res, p_res = kruskal(*groups_res)

        log.info("  raw vs residualized region ranking Spearman rho = %.3f",
                 rho_raw_vs_res)
        log.info("  top-10 overlap raw vs residualized = %d/10", overlap_top10)
        log.info("  K-W on residualized r: H=%.1f, p=%.3e", H_res, p_res)

        results["models"][name] = {
            "n_neurons": len(records),
            "ranking_raw": ranking_raw,
            "ranking_residualized": ranking_res,
            "spearman_rho_raw_vs_residualized": float(rho_raw_vs_res),
            "top10_overlap": int(overlap_top10),
            "kw_residualized": {"H": float(H_res), "p": float(p_res)},
            "per_region_mean_raw_r": means_raw,
            "per_region_mean_residualized_r": means_res,
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved: %s", OUT_PATH)


if __name__ == "__main__":
    main()
