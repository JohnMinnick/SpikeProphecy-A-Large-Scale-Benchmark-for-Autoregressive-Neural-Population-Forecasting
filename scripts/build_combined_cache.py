#!/usr/bin/env python3
"""
Build a combined Steinmetz + IBL cache for unified training.

Reads the existing Steinmetz NWB files (39 sessions) and IBL cache
(66 sessions), bins Steinmetz if needed, and merges everything into
a unified cache directory with a single M_max.

Usage:
    python scripts/build_combined_cache.py
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# --- Project root setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_combined_cache")


def main():
    # --- Paths ---
    ibl_cache = PROJECT_ROOT / "data" / "processed" / "ibl_repeated_site_cache"
    combined_cache = PROJECT_ROOT / "data" / "processed" / "combined_steinmetz_ibl_cache"
    combined_cache.mkdir(parents=True, exist_ok=True)

    # --- Load IBL metadata ---
    ibl_meta_path = ibl_cache / "metadata.json"
    with open(ibl_meta_path, "r", encoding="utf-8") as f:
        ibl_meta = json.load(f)

    logger.info(
        "IBL cache: %d sessions, M_max=%d",
        ibl_meta["num_sessions"], ibl_meta["m_max"],
    )

    # --- Process Steinmetz NWBs into uint8 .npy cache ---
    # Load the same config used for NRP Steinmetz training
    steinmetz_config_path = (
        PROJECT_ROOT / "configs" / "data" / "steinmetz_multi_nrp_50ms_no_cov.yaml"
    )
    with open(steinmetz_config_path, "r", encoding="utf-8") as f:
        steinmetz_config = yaml.safe_load(f)

    # Cache Steinmetz locally (reuses existing cache if present)
    from src.data.multi_session_loader import preprocess_and_cache

    steinmetz_cache_dir = PROJECT_ROOT / "data" / "processed" / "steinmetz_session_cache"

    logger.info("Preprocessing Steinmetz NWBs → %s", steinmetz_cache_dir)
    st_cache_path, st_meta = preprocess_and_cache(steinmetz_config)
    logger.info(
        "Steinmetz cache: %d sessions, M_max=%d",
        st_meta["num_sessions"], st_meta["m_max"],
    )

    # --- Determine global M_max ---
    global_m_max = max(st_meta["m_max"], ibl_meta["m_max"])
    logger.info("Global M_max = %d", global_m_max)

    # --- Copy and merge sessions ---
    combined_sessions = []
    combined_idx = 0

    # Steinmetz first (sessions 000 .. N_steinmetz-1)
    for sess in st_meta["sessions"]:
        src_npy = st_cache_path / f"session_{sess['index']:03d}.npy"
        counts = np.load(src_npy)  # (M_i, T_i), uint8
        m_i, t_i = counts.shape

        # Re-save as-is (no re-padding at cache level — done at load time)
        dst_npy = combined_cache / f"session_{combined_idx:03d}.npy"
        np.save(dst_npy, counts)

        combined_sessions.append({
            "index": combined_idx,
            "file": sess["file"],
            "npy_file": str(dst_npy),
            "num_units": m_i,
            "num_bins": t_i,
            "duration_s": sess["duration_s"],
            "split_boundaries": sess["split_boundaries"],
            "brain_regions": sess.get("brain_regions", None),
            "source": "steinmetz",
        })
        combined_idx += 1

        logger.info(
            "  [%03d] Steinmetz: %d units, %d bins → %s",
            combined_sessions[-1]["index"], m_i, t_i, dst_npy.name,
        )

    # IBL next (sessions N_steinmetz .. N_steinmetz+N_ibl-1)
    for sess in ibl_meta["sessions"]:
        src_npy = ibl_cache / f"session_{sess['index']:03d}.npy"
        counts = np.load(src_npy)  # (M_i, T_i), uint8
        m_i, t_i = counts.shape

        dst_npy = combined_cache / f"session_{combined_idx:03d}.npy"
        np.save(dst_npy, counts)

        combined_sessions.append({
            "index": combined_idx,
            "file": sess["file"],
            "npy_file": str(dst_npy),
            "num_units": m_i,
            "num_bins": t_i,
            "duration_s": sess["duration_s"],
            "split_boundaries": sess["split_boundaries"],
            "brain_regions": sess.get("brain_regions", None),
            "source": "ibl",
        })
        combined_idx += 1

        logger.info(
            "  [%03d] IBL: %d units, %d bins → %s",
            combined_sessions[-1]["index"], m_i, t_i, dst_npy.name,
        )

    # --- Write combined metadata ---
    all_m = [s["num_units"] for s in combined_sessions]
    combined_metadata = {
        "num_sessions": len(combined_sessions),
        "m_max": int(max(all_m)),
        "history_bins": ibl_meta.get("history_bins", 10),
        "bin_width_ms": ibl_meta.get("bin_width_ms", 50.0),
        "n_features_per_channel": 0,
        "n_covariates": 0,
        "covariate_names": [],
        "sessions": combined_sessions,
    }

    meta_out = combined_cache / "metadata.json"
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(combined_metadata, f, indent=2)

    n_st = st_meta["num_sessions"]
    n_ibl = ibl_meta["num_sessions"]
    logger.info("=" * 60)
    logger.info("COMBINED CACHE COMPLETE")
    logger.info("  Steinmetz: %d sessions", n_st)
    logger.info("  IBL:       %d sessions", n_ibl)
    logger.info("  Total:     %d sessions", len(combined_sessions))
    logger.info("  M_max:     %d", combined_metadata["m_max"])
    logger.info("  Output:    %s", combined_cache)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
