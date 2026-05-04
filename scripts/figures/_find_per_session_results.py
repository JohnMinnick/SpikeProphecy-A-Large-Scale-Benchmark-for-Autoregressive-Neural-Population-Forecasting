"""Search S3 for per-session metric outputs across the 7 architectures."""
import os, boto3, json

s3 = boto3.client(
    "s3", endpoint_url="https://s3-west.nrp-nautilus.io",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)

SLUGS = [
    ("Mamba",         "2026-03-26_baseline-mamba-v12"),
    ("Mamba_s1",      "2026-04-26_mseed-mamba-s1"),
    ("HGRN2",         "2026-04-21_2026-04-21_baseline-hgrn2-v1"),
    ("Transformer",   "2026-03-25_baseline-transformer-v12"),
    ("GatedDeltaNet", "2026-04-22_2026-04-22_baseline-gated-delta-v1"),
    ("LRU",           "2026-03-25_baseline-lru-v12"),
    ("LSTM",          "2026-04-15_baseline-lstm-v23"),
    ("SNN",           "2026-04-22_snn-standalone-3l-steinmetz"),
]

for name, slug in SLUGS:
    prefix = f"<anon>/spike-prophecy/outputs/{slug}/"
    r = s3.list_objects_v2(Bucket="braingeneersdev", Prefix=prefix, MaxKeys=50)
    print(f"\n=== {name} ({slug}) ===")
    items = r.get("Contents", [])
    if not items:
        print("  (no objects)")
        continue
    for o in items:
        # Only flag candidates likely containing per-session info
        if any(s in o["Key"].lower() for s in
               ["per_session", "metrics", "sessions", "eval", "results"]):
            print(f"  {o['Key'].replace(prefix, '')} ({o['Size']:,} B)")
        else:
            print(f"  {o['Key'].replace(prefix, '')}")
