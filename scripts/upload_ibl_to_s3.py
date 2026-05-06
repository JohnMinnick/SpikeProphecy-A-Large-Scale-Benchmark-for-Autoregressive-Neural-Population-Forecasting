"""
Upload IBL cached session arrays to <lab-bucket> S3.

Uploads the preprocessed .npy files and metadata.json from the local
IBL cache to S3, organized alongside the Steinmetz inputs:

    s3://<lab-bucket>/<anon>/spike-prophecy/inputs/steinmetz/  (existing NWBs)
    s3://<lab-bucket>/<anon>/spike-prophecy/inputs/ibl-repeated-site/
        session_000.npy
        session_001.npy
        ...
        metadata.json

Usage:
    # Dry-run (list what would be uploaded):
    python scripts/upload_ibl_to_s3.py --dry-run

    # Upload from default cache:
    python scripts/upload_ibl_to_s3.py

    # Upload from custom cache dir:
    python scripts/upload_ibl_to_s3.py --cache-dir data/processed/ibl_repeated_site_cache
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# S3 prefix structure:
#   <anon>/spike-prophecy/inputs/steinmetz/  ← existing NWB files
#   <anon>/spike-prophecy/inputs/ibl-repeated-site/  ← new IBL cache
S3_PREFIX = "<anon>/spike-prophecy/inputs/ibl-repeated-site"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload IBL cached arrays to <lab-bucket> S3.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/processed/ibl_repeated_site_cache",
        help="Local cache directory with session_NNN.npy + metadata.json.",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default=S3_PREFIX,
        help=f"S3 prefix for uploads (default: {S3_PREFIX}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be uploaded without uploading.",
    )
    return parser.parse_args()


def create_s3_client():
    """
    Create a boto3 S3 client using the EXTERNAL NRP endpoint.

    Returns:
        boto3 S3 bucket resource.
    """
    import boto3
    from botocore.config import Config

    # Retry config for reliability
    s3_config = Config(
        retries={"max_attempts": 5, "mode": "adaptive"},
        connect_timeout=30,
        read_timeout=120,
    )

    s3 = boto3.resource(
        "s3",
        # External endpoint for local machine access
        endpoint_url="https://s3-west.nrp-nautilus.io",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=s3_config,
    )
    return s3.Bucket("<lab-bucket>")


def main() -> None:
    """Upload IBL cache to S3."""
    args = parse_args()
    cache_dir = Path(args.cache_dir)

    if not cache_dir.exists():
        print(f"ERROR: Cache directory not found: {cache_dir}")
        sys.exit(1)

    # Gather files to upload: all .npy + metadata.json
    npy_files = sorted(cache_dir.glob("session_*.npy"))
    metadata_file = cache_dir / "metadata.json"

    files_to_upload = list(npy_files)
    if metadata_file.exists():
        files_to_upload.append(metadata_file)

    if not files_to_upload:
        print("No files found to upload.")
        sys.exit(0)

    # Calculate totals
    total_bytes = sum(f.stat().st_size for f in files_to_upload)
    total_gb = total_bytes / (1024 ** 3)

    # Print metadata summary if available
    if metadata_file.exists():
        with open(metadata_file) as f:
            meta = json.load(f)
        print(f"\nIBL Cache Summary:")
        print(f"  Sessions: {meta.get('num_sessions', '?')}")
        print(f"  M_max: {meta.get('m_max', '?')}")
        print(f"  Bin width: {meta.get('bin_width_ms', '?')}ms")

    print(f"\n{'='*60}")
    print(f"Uploading {len(files_to_upload)} files to S3")
    print(f"  Local dir:  {cache_dir}")
    print(f"  S3 prefix:  s3://<lab-bucket>/{args.s3_prefix}/")
    print(f"  Total size: {total_gb:.2f} GB")
    print(f"  Dry run:    {args.dry_run}")
    print(f"{'='*60}\n")

    if args.dry_run:
        for f in files_to_upload:
            size_mb = f.stat().st_size / (1024 ** 2)
            s3_key = f"{args.s3_prefix}/{f.name}"
            print(f"  [DRY RUN] {f.name} ({size_mb:.1f} MB) -> s3://{s3_key}")
        print(f"\nDry run complete. {len(files_to_upload)} files would be uploaded.")
        return

    # Upload
    bucket = create_s3_client()
    uploaded = 0
    failed = 0

    for i, filepath in enumerate(files_to_upload, 1):
        s3_key = f"{args.s3_prefix}/{filepath.name}"
        size_mb = filepath.stat().st_size / (1024 ** 2)
        print(f"  [{i}/{len(files_to_upload)}] {filepath.name} ({size_mb:.1f} MB)", end="")

        try:
            bucket.upload_file(str(filepath), s3_key)
            print(" OK")
            uploaded += 1
        except Exception as e:
            print(f" FAILED: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Upload complete: {uploaded} succeeded, {failed} failed")
    print(f"S3 location: s3://<lab-bucket>/{args.s3_prefix}/")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
