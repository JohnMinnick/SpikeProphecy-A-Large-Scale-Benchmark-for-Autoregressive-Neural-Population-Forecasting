"""Region-hierarchy sensitivity check.

Addresses two reviewer concerns about the Kruskal-Wallis claim in §4.5
of docs/neurips_neurocog/main.tex:

  1. With n=27,144 neurons, p < 10^-300 reflects sample size more than
     effect size. We additionally report eta-squared (epsilon-squared
     variant for K-W: eta^2 = (H - k + 1) / (n - k)).
  2. K-W assumes independent observations, but neurons within a session
     share recording conditions (probe geometry, animal state, day).
     We test rank-order stability via leave-one-session-out (LOSO):
     for each held-out session, recompute per-region mean r from the
     remaining 38 sessions, and report Spearman rho against the
     full-data ranking.

Usage:
    python scripts/eval_region_hierarchy_sensitivity.py
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
    "1L SNN": PROJECT_ROOT / "outputs" / "eval_local" / "multihead_1l_v3_full.json",
    "2L SNN": PROJECT_ROOT / "outputs" / "eval_local" / "multihead_2l_v3_full.json",
}
REGION_MAP_PATH = (
    PROJECT_ROOT / "outputs" / "eval_analysis" / "brain_region_mapping.json"
)
OUT_PATH = (
    PROJECT_ROOT / "outputs" / "eval_local"
    / "region_hierarchy_sensitivity.json"
)
MIN_NEURONS_PER_REGION = 30  # match the main analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("region_sensitivity")


def load_per_neuron_by_session(eval_path: Path, region_map: dict):
    """Return dict[session_idx] -> list of (region, r) tuples (Steinmetz only)."""
    d = json.load(open(eval_path))
    sessions_map = region_map["sessions"]
    out = {}
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
        n = min(len(regions), len(r_list))
        out[sidx] = [(regions[i], float(r_list[i])) for i in range(n)]
    return out


def per_region_arrays(by_session: dict, exclude: int | None = None):
    """Stack per-neuron r values by region across all sessions (optional excl)."""
    out = defaultdict(list)
    for sidx, pairs in by_session.items():
        if exclude is not None and sidx == exclude:
            continue
        for region, r in pairs:
            out[region].append(r)
    return {rg: np.array(rs) for rg, rs in out.items()}


def compute_kw_and_eta(arrays: dict):
    """K-W H, p, n, k, eta^2 over regions with >= MIN_NEURONS_PER_REGION."""
    groups = [v for v in arrays.values() if len(v) >= MIN_NEURONS_PER_REGION]
    if len(groups) < 2:
        return None
    H, p = kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    # epsilon-squared form of eta-squared for K-W (Tomczak & Tomczak 2014)
    eta2 = (H - k + 1) / (n - k) if (n - k) > 0 else float("nan")
    return {
        "H": float(H),
        "p": float(p),
        "n_neurons": int(n),
        "n_regions": int(k),
        "eta_squared": float(eta2),
    }


def region_ranking(arrays: dict, min_n: int = MIN_NEURONS_PER_REGION):
    """Return list of regions sorted by mean r (descending), filtered by min_n."""
    means = {rg: float(np.mean(v)) for rg, v in arrays.items() if len(v) >= min_n}
    sorted_regions = sorted(means.keys(), key=lambda r: -means[r])
    return sorted_regions, means


def loso_spearman(by_session: dict):
    """Leave-one-session-out: Spearman rho vs full-data ranking, on common regions."""
    # full-data ranking
    full_arrays = per_region_arrays(by_session)
    full_sorted, full_means = region_ranking(full_arrays)
    # rank lookup
    full_rank = {rg: i for i, rg in enumerate(full_sorted)}

    rhos = []
    n_common_list = []
    for sidx in sorted(by_session.keys()):
        loso_arrays = per_region_arrays(by_session, exclude=sidx)
        loso_sorted, loso_means = region_ranking(loso_arrays)
        # restrict to regions present in BOTH rankings
        common = [r for r in loso_sorted if r in full_rank]
        if len(common) < 5:
            continue
        loso_rank = {rg: i for i, rg in enumerate(loso_sorted) if rg in full_rank}
        x = [full_rank[r] for r in common]
        y = [loso_rank[r] for r in common]
        rho, _ = spearmanr(x, y)
        rhos.append(float(rho))
        n_common_list.append(len(common))
    return {
        "n_folds": len(rhos),
        "rho_mean": float(np.mean(rhos)),
        "rho_std": float(np.std(rhos)),
        "rho_min": float(np.min(rhos)),
        "rho_max": float(np.max(rhos)),
        "rho_p25": float(np.percentile(rhos, 25)),
        "rho_median": float(np.median(rhos)),
        "rho_p75": float(np.percentile(rhos, 75)),
        "n_regions_common_mean": float(np.mean(n_common_list)),
        "rhos": rhos,
    }


def main():
    log.info("Loading region map: %s", REGION_MAP_PATH)
    region_map = json.load(open(REGION_MAP_PATH))

    results = {
        "min_neurons_per_region": MIN_NEURONS_PER_REGION,
        "models": {},
    }

    for name, path in EVAL_JSONS.items():
        log.info("=" * 60)
        log.info("Model: %s", name)
        log.info("Loading eval: %s", path)
        by_session = load_per_neuron_by_session(path, region_map)
        log.info("Steinmetz sessions: %d", len(by_session))

        full_arrays = per_region_arrays(by_session)
        kw = compute_kw_and_eta(full_arrays)
        log.info(
            "  K-W H=%.1f, p=%.3e, n=%d, k=%d, eta^2=%.4f",
            kw["H"], kw["p"], kw["n_neurons"], kw["n_regions"], kw["eta_squared"],
        )

        loso = loso_spearman(by_session)
        log.info(
            "  LOSO Spearman rho: mean=%.3f, min=%.3f, "
            "median=%.3f (n=%d folds, ~%.0f common regions)",
            loso["rho_mean"], loso["rho_min"], loso["rho_median"],
            loso["n_folds"], loso["n_regions_common_mean"],
        )

        results["models"][name] = {
            "kruskal_wallis": kw,
            "loso_spearman": loso,
        }

    # Cross-architecture: compare 1L vs 2L full ranking
    log.info("=" * 60)
    log.info("Cross-architecture rank consistency (1L vs 2L full data)")
    by_1l = load_per_neuron_by_session(EVAL_JSONS["1L SNN"], region_map)
    by_2l = load_per_neuron_by_session(EVAL_JSONS["2L SNN"], region_map)
    arr_1l = per_region_arrays(by_1l)
    arr_2l = per_region_arrays(by_2l)
    sorted_1l, means_1l = region_ranking(arr_1l)
    sorted_2l, means_2l = region_ranking(arr_2l)
    common = [r for r in sorted_1l if r in means_2l]
    rank_1l = {r: i for i, r in enumerate(sorted_1l)}
    rank_2l = {r: i for i, r in enumerate(sorted_2l)}
    x = [rank_1l[r] for r in common]
    y = [rank_2l[r] for r in common]
    rho_xarch, _ = spearmanr(x, y)
    log.info("  cross-arch Spearman rho = %.3f (%d regions)", rho_xarch, len(common))
    results["cross_architecture"] = {
        "spearman_rho": float(rho_xarch),
        "n_regions": int(len(common)),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved: %s", OUT_PATH)


if __name__ == "__main__":
    main()
