#!/usr/bin/env python3
"""
Resume IBL caching for the 13 remaining sessions (indices 53-65).

Uses concurrent.futures to enforce a hard per-session timeout so that
hanging ONE API calls don't block the entire pipeline.

Usage:
    python scripts/resume_ibl_cache.py [--timeout 300]
"""
import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
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
logger = logging.getLogger("resume_ibl_cache")


def _process_single_session(eid: str, config: dict, one_cache_dir: str,
                            bin_width_ms: float) -> dict:
    """
    Process one IBL session in an isolated process.

    Returns a dict with session stats, or raises on failure.
    This runs in a subprocess so it can be killed by timeout.
    """
    # Re-import inside subprocess to avoid pickle issues
    import numpy as np
    from src.data.ibl_data_loader import load_ibl_session
    from src.data.binning import bin_spike_trains

    sorting, load_meta = load_ibl_session(
        eid, config, cache_dir=one_cache_dir,
    )

    counts, bin_meta = bin_spike_trains(
        sorting, bin_width_ms=bin_width_ms,
    )
    m_i, t_i = counts.shape

    # Clamp to uint8
    overflow = int(np.sum(counts > 255))
    counts_u8 = np.clip(counts, 0, 255).astype(np.uint8)

    return {
        "counts": counts_u8,
        "num_units": m_i,
        "num_bins": t_i,
        "overflow": overflow,
        "brain_regions": load_meta.get("brain_regions", None),
    }


def main():
    parser = argparse.ArgumentParser(description="Resume IBL cache for remaining sessions")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-session timeout in seconds (default: 300)")
    args = parser.parse_args()

    # --- Load config ---
    config_path = PROJECT_ROOT / "configs" / "data" / "ibl_repeated_site_full_nrp.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    bin_width_ms = config.get("bin_width_ms", 50.0)
    splits = config.get("splits", {"train": 0.7, "val": 0.15, "test": 0.15})
    ibl_cfg = config.get("ibl", {})
    one_cache_dir = ibl_cfg.get("cache_dir", "data/raw/ibl")

    cache_path = PROJECT_ROOT / "data" / "processed" / "ibl_repeated_site_cache"

    # --- Load existing metadata ---
    metadata_file = cache_path / "metadata.json"
    with open(metadata_file, "r", encoding="utf-8") as f:
        existing_meta = json.load(f)

    existing_count = existing_meta["num_sessions"]
    existing_eids = {s["file"] for s in existing_meta["sessions"]}
    logger.info("Existing cache: %d sessions", existing_count)

    # --- Load all 66 successfully downloaded EIDs ---
    eids_file = PROJECT_ROOT / "data" / "raw" / "ibl" / "session_eids.json"
    with open(eids_file, "r", encoding="utf-8") as f:
        eids_data = json.load(f)

    all_eids = eids_data["eids"]  # 66 valid EIDs
    failed_eids = {d["eid"] for d in eids_data.get("failed_details", [])}

    # --- Identify remaining EIDs to process ---
    remaining = [(i, eid) for i, eid in enumerate(all_eids)
                 if eid not in existing_eids and eid not in failed_eids]

    logger.info("Remaining sessions to process: %d", len(remaining))
    if not remaining:
        logger.info("Nothing to do — all sessions already cached!")
        return

    # --- Process each remaining session with timeout ---
    new_sessions = []
    skipped = []
    next_index = existing_count  # Continue numbering from where we left off

    for orig_idx, eid in remaining:
        logger.info(
            "[%d/%d] Processing session %s (timeout=%ds)...",
            next_index + 1, existing_count + len(remaining), eid, args.timeout,
        )

        t0 = time.time()
        try:
            # Use ProcessPoolExecutor for hard kill on timeout
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _process_single_session,
                    eid, config, one_cache_dir, bin_width_ms,
                )
                result = future.result(timeout=args.timeout)

            elapsed = time.time() - t0
            counts_u8 = result["counts"]
            m_i = result["num_units"]
            t_i = result["num_bins"]

            # Save .npy file
            npy_path = cache_path / f"session_{next_index:03d}.npy"
            np.save(npy_path, counts_u8)

            # Compute split boundaries
            train_end = int(t_i * splits["train"])
            val_end = train_end + int(t_i * splits["val"])

            session_entry = {
                "index": next_index,
                "file": eid,
                "npy_file": f"data\\processed\\ibl_repeated_site_cache\\{npy_path.name}",
                "num_units": m_i,
                "num_bins": t_i,
                "duration_s": round(t_i * bin_width_ms / 1000, 2),
                "split_boundaries": {
                    "train_end": train_end,
                    "val_end": val_end,
                },
                "brain_regions": result["brain_regions"],
            }
            new_sessions.append(session_entry)
            next_index += 1

            logger.info(
                "  ✓ Session %03d: %d units, %d bins (%.1fs) — processed in %.1fs",
                session_entry["index"], m_i, t_i,
                t_i * bin_width_ms / 1000, elapsed,
            )

            if result["overflow"] > 0:
                logger.warning("  ⚠ %d bins exceeded uint8 range", result["overflow"])

        except FuturesTimeout:
            elapsed = time.time() - t0
            logger.error(
                "  ✗ TIMEOUT after %.0fs for session %s — skipping", elapsed, eid,
            )
            skipped.append({"eid": eid, "reason": f"timeout ({args.timeout}s)"})

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(
                "  ✗ FAILED after %.0fs for session %s: %s — skipping",
                elapsed, eid, e,
            )
            skipped.append({"eid": eid, "reason": str(e)})

    # --- Merge into existing metadata ---
    all_sessions = existing_meta["sessions"] + new_sessions
    all_m = [s["num_units"] for s in all_sessions]
    m_max = int(max(all_m))

    final_meta = {
        "num_sessions": len(all_sessions),
        "m_max": m_max,
        "history_bins": config.get("history_bins", 10),
        "bin_width_ms": bin_width_ms,
        "n_features_per_channel": 0,
        "n_covariates": 0,
        "covariate_names": [],
        "sessions": all_sessions,
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(final_meta, f, indent=2)

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("RESUME COMPLETE")
    logger.info("  New sessions added: %d", len(new_sessions))
    logger.info("  Skipped (timeout/error): %d", len(skipped))
    logger.info("  Total cached sessions: %d", len(all_sessions))
    logger.info("  M_max: %d", m_max)
    logger.info("  metadata.json updated: %s", metadata_file)
    logger.info("=" * 60)

    if skipped:
        logger.warning("Skipped sessions:")
        for s in skipped:
            logger.warning("  %s — %s", s["eid"], s["reason"])


if __name__ == "__main__":
    main()
