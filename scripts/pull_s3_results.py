"""
Pull experiment results from S3 for local analysis.

Downloads metrics.json (and optionally plots) from the braingeneers S3
bucket into outputs/s3_metrics/ for paper figure generation and analysis.

Usage:
    python scripts/pull_s3_results.py                        # all experiments
    python scripts/pull_s3_results.py --experiment sweep2    # filter by name
    python scripts/pull_s3_results.py --plots                # include plots
    python scripts/pull_s3_results.py --list                 # list only

Environment variables:
    AWS_ACCESS_KEY_ID     — S3 access key
    AWS_SECRET_ACCESS_KEY — S3 secret key
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import boto3
from botocore.config import Config


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
S3_ENDPOINT = "https://s3-west.nrp-nautilus.io"
BUCKET = "braingeneersdev"
S3_PREFIX = "<anon>/spike-prophecy/outputs"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "s3_metrics"


def create_s3_client():
    """
    Create a boto3 S3 client using the external NRP endpoint.

    Reads credentials from environment variables.

    Returns:
        boto3 S3 client configured for the braingeneers bucket.
    """
    s3_config = Config(
        retries={"max_attempts": 3, "mode": "standard"},
        connect_timeout=10,
        read_timeout=30,
    )
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=s3_config,
    )


def list_experiments(s3_client, filter_name=None):
    """
    List all experiment folders and their files on S3.

    Args:
        s3_client: boto3 S3 client.
        filter_name: Optional substring to filter experiment names.

    Returns:
        Dict mapping experiment name to list of (key, size, modified) tuples.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=BUCKET, Prefix=S3_PREFIX)

    experiments = defaultdict(list)
    for response in pages:
        if "Contents" not in response:
            continue
        for obj in response["Contents"]:
            key = obj["Key"]
            # Key format: <anon>/spike-prophecy/outputs/<exp_name>/<file>
            relative = key[len(S3_PREFIX) + 1:]  # strip prefix + /
            parts = relative.split("/", 1)
            if len(parts) < 2:
                continue
            exp_name = parts[0]
            filename = parts[1]

            # Apply filter if specified
            if filter_name and filter_name not in exp_name:
                continue

            experiments[exp_name].append({
                "key": key,
                "filename": filename,
                "size": obj["Size"],
                "modified": str(obj["LastModified"]),
            })

    return experiments


def download_experiment_files(
    s3_client, experiments, output_dir, include_plots=False,
):
    """
    Download experiment files from S3 to local output directory.

    Args:
        s3_client: boto3 S3 client.
        experiments: Dict from list_experiments().
        output_dir: Local directory to save files.
        include_plots: If True, also download plots/*.png files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # File patterns to download
    target_patterns = [
        "metrics.json",
        "config.yaml",
        "RUN.md",
        "notes.md",
    ]
    if include_plots:
        target_patterns.extend([".png", ".pdf"])

    total_downloaded = 0

    for exp_name in sorted(experiments.keys()):
        files = experiments[exp_name]
        exp_dir = output_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        for file_info in files:
            filename = file_info["filename"]
            # Check if this file matches any target pattern
            should_download = any(
                filename == pat or filename.endswith(pat)
                for pat in target_patterns
            )
            if not should_download:
                continue

            # Create subdirectory structure (e.g., plots/)
            local_path = exp_dir / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"  Downloading {exp_name}/{filename} "
                  f"({file_info['size']:,} bytes)")
            s3_client.download_file(BUCKET, file_info["key"], str(local_path))
            total_downloaded += 1

    print(f"\nDownloaded {total_downloaded} files to {output_dir}")
    return total_downloaded


def print_summary(experiments):
    """
    Print a summary table of all experiments on S3.

    Args:
        experiments: Dict from list_experiments().
    """
    print(f"\n{'Experiment':<50} {'Files':>6} {'Has Metrics':>12} "
          f"{'Size (MB)':>10}")
    print("-" * 82)

    for exp_name in sorted(experiments.keys()):
        files = experiments[exp_name]
        total_size = sum(f["size"] for f in files)
        has_metrics = any(
            f["filename"] == "metrics.json" for f in files
        )
        print(f"{exp_name:<50} {len(files):>6} "
              f"{'Yes' if has_metrics else 'NO':>12} "
              f"{total_size / 1e6:>10.1f}")

    print(f"\nTotal: {len(experiments)} experiments")


def main():
    """Main entrypoint: list or download S3 experiment results."""
    parser = argparse.ArgumentParser(
        description="Pull experiment results from S3 for local analysis.",
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Filter by experiment name substring.",
    )
    parser.add_argument(
        "--plots", action="store_true",
        help="Also download plot files (PNG/PDF).",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_only",
        help="List experiments only, don't download.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Local directory for downloaded files.",
    )
    args = parser.parse_args()

    # Verify credentials
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        print("ERROR: AWS_ACCESS_KEY_ID not set. "
              "Set S3 credentials in environment variables.")
        sys.exit(1)

    print(f"Connecting to S3 ({S3_ENDPOINT})...")
    s3_client = create_s3_client()

    print(f"Listing experiments under {S3_PREFIX}...")
    experiments = list_experiments(s3_client, filter_name=args.experiment)

    if not experiments:
        print("No experiments found.")
        return

    print_summary(experiments)

    if args.list_only:
        return

    print(f"\nDownloading to {args.output_dir}...")
    download_experiment_files(
        s3_client, experiments, args.output_dir,
        include_plots=args.plots,
    )

    # Print quick metrics summary for experiments that have metrics.json
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    output_dir = Path(args.output_dir)
    for exp_name in sorted(experiments.keys()):
        metrics_path = output_dir / exp_name / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            val_r = m.get("teacher_best_val_pearson_r", "N/A")
            val_loss = m.get("teacher_best_val_loss", "N/A")
            epochs = m.get("teacher_n_epochs_trained", "N/A")
            if isinstance(val_r, float):
                print(f"  {exp_name}: val_r={val_r:.4f}, "
                      f"val_loss={val_loss:.4f}, epochs={epochs}")
            else:
                print(f"  {exp_name}: {val_r}")


if __name__ == "__main__":
    main()
