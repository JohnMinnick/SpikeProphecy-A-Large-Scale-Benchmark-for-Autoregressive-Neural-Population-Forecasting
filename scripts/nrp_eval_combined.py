"""
NRP combined-dataset evaluation wrapper.

Downloads the combined Steinmetz+IBL cache from S3, then runs
nrp_recalc_eval.py for corrected per-session weighted-r metrics.

This script handles the IBL data download that nrp_recalc_eval.py
doesn't support natively.

Usage (NRP):
    python scripts/nrp_eval_combined.py \
        --checkpoint-slug multihead-1l-v3 \
        --data-config configs/data/combined_steinmetz_ibl_nrp.yaml \
        --student-config configs/student/standalone_multihead_1l.yaml \
        --teacher-val-r 0.543
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nrp_eval_combined")


def download_combined_cache(data_config: dict) -> Path:
    """
    Download the combined Steinmetz+IBL cache from S3.

    Mirrors the download logic from train_snn_standalone.py lines 472-546.
    Downloads both:
      1. IBL/combined pre-cached .npy files from S3
      2. Steinmetz NWB files from S3

    Returns:
        Path to the local cache directory.
    """
    source_type = data_config.get("source", {}).get("type", "nwb_multi")

    # Add nrp/ to path for s3_utils
    nrp_dir = PROJECT_ROOT / "nrp"
    sys.path.insert(0, str(nrp_dir))
    from s3_utils import list_files, download_single_file

    if source_type == "ibl":
        # IBL or combined data: download pre-cached session arrays
        ibl_tag = data_config.get("ibl", {}).get("tag", "repeated_site")
        tag_to_prefix = {
            "repeated_site": "<anon>/spike-prophecy/inputs/ibl-repeated-site",
            "combined": "<anon>/spike-prophecy/inputs/combined-steinmetz-ibl",
        }
        tag_to_cache = {
            "repeated_site": "ibl_repeated_site_cache",
            "combined": "combined_steinmetz_ibl_cache",
        }
        default_prefix = tag_to_prefix.get(
            ibl_tag, f"<anon>/spike-prophecy/inputs/{ibl_tag}"
        )
        default_cache = tag_to_cache.get(ibl_tag, f"{ibl_tag}_cache")

        ibl_s3_prefix = os.environ.get("S3_IBL_PREFIX", default_prefix)
        ibl_cache_name = os.environ.get("S3_IBL_CACHE_DIR", default_cache)
        ibl_cache = PROJECT_ROOT / "data" / "processed" / ibl_cache_name
        ibl_cache.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Downloading combined cache from S3 prefix: %s", ibl_s3_prefix
        )

        all_keys = list_files(ibl_s3_prefix)
        ibl_keys = [
            k for k in all_keys
            if k.endswith(".npy") or k.endswith(".json")
        ]
        logger.info(
            "Found %d cache files in S3 under %s",
            len(ibl_keys), ibl_s3_prefix,
        )

        for key in ibl_keys:
            filename = os.path.basename(key)
            local_path = str(ibl_cache / filename)
            if not Path(local_path).exists():
                logger.info("  Downloading %s", filename)
                download_single_file(key=key, local_path=local_path)
            else:
                logger.info("  Already cached: %s", filename)

        # Verify metadata
        meta_path = ibl_cache / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            logger.info(
                "Cache ready: %d sessions, M_max=%d",
                meta.get("num_sessions", 0),
                meta.get("m_max", 0),
            )
        else:
            logger.warning("metadata.json not found in cache!")

        return ibl_cache

    else:
        # Steinmetz-only: download NWB files
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "nrp_train", PROJECT_ROOT / "scripts" / "nrp_train.py"
        )
        nrp_train = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nrp_train)
        nrp_train.download_nwb_from_s3()
        return PROJECT_ROOT / "data" / "processed" / "session_cache"


def main():
    """Download combined data then run nrp_recalc_eval.py."""
    parser = argparse.ArgumentParser(
        description="Combined dataset eval wrapper for NRP.",
    )
    parser.add_argument(
        "--checkpoint-slug", type=str, required=True,
        help="S3 experiment slug for the checkpoint.",
    )
    parser.add_argument(
        "--data-config", type=str, required=True,
        help="Path to data config YAML.",
    )
    parser.add_argument(
        "--student-config", type=str, required=True,
        help="Path to student config YAML.",
    )
    parser.add_argument(
        "--teacher-val-r", type=float, default=0.543,
        help="Teacher val_r for retention calculation.",
    )
    args = parser.parse_args()

    # Step 1: Load data config and download combined cache
    from src.utils.config import load_config
    data_config = load_config(args.data_config)

    logger.info("=" * 60)
    logger.info("COMBINED EVAL WRAPPER — downloading data first")
    logger.info("=" * 60)

    download_combined_cache(data_config)

    # Step 2: Also download Steinmetz NWB if this is a combined config
    # (nrp_recalc_eval needs them for the NWB sessions)
    ibl_tag = data_config.get("ibl", {}).get("tag", "")
    if ibl_tag == "combined":
        logger.info("Combined config — also downloading Steinmetz NWB files...")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "nrp_train", PROJECT_ROOT / "scripts" / "nrp_train.py"
        )
        nrp_train = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nrp_train)
        nrp_train.download_nwb_from_s3()

    # Step 3: Run the actual eval
    logger.info("=" * 60)
    logger.info("Data download complete — launching nrp_recalc_eval.py")
    logger.info("=" * 60)

    import subprocess
    result = subprocess.run(
        [
            sys.executable, "scripts/nrp_recalc_eval.py",
            "--checkpoint-slug", args.checkpoint_slug,
            "--data-config", args.data_config,
            "--student-config", args.student_config,
            "--teacher-val-r", str(args.teacher_val_r),
        ],
        cwd=str(PROJECT_ROOT),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
