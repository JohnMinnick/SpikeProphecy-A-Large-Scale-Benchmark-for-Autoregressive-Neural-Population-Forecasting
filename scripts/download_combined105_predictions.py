"""Download Mamba combined-105 predictions from S3 to local.

After deploy_save_pred_combined105.ps1's NRP job uploads
session_NNN.npz files to
``s3://braingeneersdev/jrm/spike-prophecy/outputs/save-pred-combined105-mamba/predictions/``,
this script pulls them locally to
``outputs/eval_local/behavioral_predictions/mamba_combined105/`` so
the eval scripts can read them.
"""
import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--s3-prefix",
        default="jrm/spike-prophecy/outputs/save-pred-combined105-mamba/predictions_full/",
    )
    p.add_argument(
        "--local-dir",
        default="outputs/eval_local/behavioral_predictions/mamba_combined105",
    )
    p.add_argument(
        "--endpoint-url",
        default="https://s3-west.nrp-nautilus.io",
    )
    args = p.parse_args()

    import boto3  # type: ignore
    s3 = boto3.client(
        "s3",
        endpoint_url=args.endpoint_url,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    bucket = "braingeneersdev"
    out_dir = Path(args.local_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    n_dl = 0
    n_skip = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=args.s3_prefix):
        for obj in page.get("Contents", []):
            fn = obj["Key"].split("/")[-1]
            if not fn.endswith(".npz"):
                continue
            local = out_dir / fn
            if local.exists() and local.stat().st_size == obj["Size"]:
                n_skip += 1
                continue
            print(f"  fetching {fn} ({obj['Size'] / 1e6:.1f} MB)")
            s3.download_file(bucket, obj["Key"], str(local))
            n_dl += 1
    print(f"Downloaded {n_dl} new files, skipped {n_skip} cached.")


if __name__ == "__main__":
    main()
