"""
Pull population metrics results from S3 and update Table 1 in the NeurIPS paper.

Usage:
    python scripts/pull_pop_metrics.py
"""
import json
import boto3
from botocore.config import Config as BotoConfig
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
S3_BUCKET = "braingeneersdev"
S3_PREFIX = "<anon>/spike-prophecy/outputs"

# S3 slugs for each architecture's population metrics
SLUGS = {
    "mamba": "teacher-pop-metrics-steinmetz",
    "transformer": "pop-metrics-transformer-steinmetz",
    "lru": "pop-metrics-lru-steinmetz",
    # "lstm": "pop-metrics-lstm-steinmetz",  # TODO: add when available
}

# Local output directory
OUT_DIR = Path("outputs/pop_metrics_table1")


def get_s3():
    """Create S3 client (external endpoint for local use)."""
    return boto3.client("s3",
        endpoint_url="https://s3-west.nrp-nautilus.io",
        config=BotoConfig(retries={"max_attempts": 3}))


def pull_metrics(s3, slug):
    """Download pop_metrics.json from S3 and return parsed dict."""
    key = f"{S3_PREFIX}/{slug}/pop_metrics.json"
    try:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except Exception as e:
        print(f"  WARN {slug}: {e}")
        return None


def main():
    """Pull all available population metrics and print Table 1 values."""
    s3 = get_s3()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SpikeProphecy Table 1 — Population Metrics Pull")
    print("=" * 70)

    # Collect all results
    results = {}
    for arch, slug in SLUGS.items():
        print(f"\nPulling {arch} ({slug})...")
        data = pull_metrics(s3, slug)
        if data is not None:
            results[arch] = data
            # Save locally
            local_path = OUT_DIR / f"{arch}_pop_metrics.json"
            with open(local_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  OK: {data['n_sessions']} sessions, "
                  f"{data['total_neurons']:,} neurons")
        else:
            print(f"  PENDING")

    # Print Table 1 format
    print("\n" + "=" * 70)
    print("TABLE 1 VALUES (LaTeX-ready)")
    print("=" * 70)
    print(f"{'Model':<15} {'Params':>8} {'Wt-r':>8} {'Pop Rate r':>12} "
          f"{'Spatial r':>12} {'Cosine':>12} {'MAE':>8}")
    print("-" * 70)

    for arch, data in results.items():
        avg = data["weighted_avg"]
        params = f"{data['n_params']:,}" if "n_params" in data else "---"
        print(f"{arch:<15} {params:>8} {'---':>8} "
              f"{avg.get('pop_rate_r', '---'):>12} "
              f"{avg.get('spatial_r', '---'):>12} "
              f"{avg.get('cosine_sim', '---'):>12} "
              f"{'---':>8}")

    # Print LaTeX table rows
    print("\n--- LaTeX rows ---")
    for arch, data in results.items():
        avg = data["weighted_avg"]
        pr = avg.get("pop_rate_r", "---")
        sr = avg.get("spatial_r", "---")
        cs = avg.get("cosine_sim", "---")
        print(f"    {arch.title()}")
        print(f"      & PARAMS & WT_R & {pr} & {sr}")
        print(f"      & {cs} & MAE \\\\")


if __name__ == "__main__":
    main()
