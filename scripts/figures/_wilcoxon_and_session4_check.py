"""Compute Wilcoxon paired tests on the cluster-vs-LSTM/SNN gap, and
identify the most representative session for the hero figure.

Outputs:
  - Per-architecture per-session r values
  - Wilcoxon paired test p-values: each cluster member vs LSTM and SNN
  - Session 4 specific: where it sits in the per-session r distribution

Saves a JSON summary to data/figure_cache/wilcoxon_summary.json so the
paper can cite specific p-values without re-running.
"""

import os
import json
from pathlib import Path
import boto3
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "figure_cache" / "wilcoxon_summary.json"

# Per-architecture S3 metrics keys (Steinmetz 39-session evals)
SLUGS = [
    ("Mamba",         "teacher-pop-metrics-steinmetz"),
    ("HGRN2",         "teacher-pop-metrics-hgrn2-steinmetz"),
    ("Transformer",   "pop-metrics-transformer-steinmetz"),
    ("GatedDeltaNet", "teacher-pop-metrics-gated-delta-steinmetz"),
    ("LRU",           "pop-metrics-lru-steinmetz"),
    ("LSTM",          "pop-metrics-lstm-steinmetz"),
    ("SNN (3L)",      "pop-metrics-snn-3l-steinmetz"),
]
CLUSTER = ["Mamba", "HGRN2", "Transformer", "GatedDeltaNet", "LRU"]
TRAILING = ["LSTM", "SNN (3L)"]

s3 = boto3.client(
    "s3", endpoint_url="https://s3-west.nrp-nautilus.io",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)


def fetch_per_session(slug):
    key = f"jrm/spike-prophecy/outputs/{slug}/pop_metrics.json"
    obj = s3.get_object(Bucket="braingeneersdev", Key=key)
    data = json.loads(obj["Body"].read())
    return data["per_session"]


def main():
    # Build per-architecture, per-session arrays
    per_arch = {}
    for name, slug in SLUGS:
        rows = fetch_per_session(slug)
        rows_by_idx = {r["session_idx"]: r for r in rows}
        per_arch[name] = rows_by_idx
        print(f"{name:14s}: {len(rows)} sessions, sample r="
              f"{rows[0].get('pearson_r'):.4f}")

    # Find common session indices
    sess_sets = [set(d.keys()) for d in per_arch.values()]
    common = sorted(set.intersection(*sess_sets))
    print(f"\nCommon sessions across all 7 archs: n={len(common)}")
    print(f"Indices: {common[:5]}...{common[-3:]}")

    # Build Pearson r arrays per arch
    metrics = ["pearson_r", "pop_rate_r", "spatial_r", "cosine_sim"]
    arrays = {m: {} for m in metrics}
    for name, rows in per_arch.items():
        for m in metrics:
            arrays[m][name] = np.array(
                [rows[i].get(m, np.nan) for i in common], dtype=np.float64
            )

    # Per-arch summary
    summary = {"n_sessions": len(common), "session_indices": common,
               "per_arch": {}, "wilcoxon": {}}
    print("\n=== Per-architecture r summary (common 39 sessions) ===")
    print(f"{'Arch':14s} {'mean r':>8s} {'sd':>8s} {'median':>8s}")
    for name in [n for n, _ in SLUGS]:
        a = arrays["pearson_r"][name]
        summary["per_arch"][name] = {
            "mean_r": float(np.mean(a)),
            "sd_r": float(np.std(a, ddof=1)),
            "median_r": float(np.median(a)),
            "se_r": float(np.std(a, ddof=1) / np.sqrt(len(a))),
        }
        print(f"{name:14s} {np.mean(a):8.4f} {np.std(a, ddof=1):8.4f} "
              f"{np.median(a):8.4f}")

    # Wilcoxon paired tests: each cluster member vs LSTM and SNN
    print("\n=== Wilcoxon paired tests (Steinmetz 39, two-sided) ===")
    print(f"{'Comparison':35s} {'mean diff':>10s} {'W':>10s} {'p':>12s}")
    for cluster_name in CLUSTER:
        for trailing_name in TRAILING:
            a = arrays["pearson_r"][cluster_name]
            b = arrays["pearson_r"][trailing_name]
            d = a - b
            W, p = wilcoxon(d, alternative="two-sided")
            key = f"{cluster_name} vs {trailing_name}"
            summary["wilcoxon"][key] = {
                "mean_diff_r": float(np.mean(d)),
                "median_diff_r": float(np.median(d)),
                "W": float(W),
                "p_two_sided": float(p),
                "n_pairs": int(len(d)),
            }
            print(f"{key:35s} {np.mean(d):10.4f} {W:10.1f} {p:12.2e}")

    # Cluster-min vs LSTM/SNN: pick the worst cluster member per session,
    # test against trailing — the strongest test for the gap claim
    cluster_arr = np.stack([arrays["pearson_r"][n] for n in CLUSTER], axis=0)
    cluster_min = cluster_arr.min(axis=0)
    print("\n=== Cluster-MIN (worst-of-5 per session) vs trailing ===")
    for trailing_name in TRAILING:
        b = arrays["pearson_r"][trailing_name]
        d = cluster_min - b
        W, p = wilcoxon(d, alternative="two-sided")
        key = f"cluster-min vs {trailing_name}"
        summary["wilcoxon"][key] = {
            "mean_diff_r": float(np.mean(d)),
            "median_diff_r": float(np.median(d)),
            "W": float(W),
            "p_two_sided": float(p),
            "n_pairs": int(len(d)),
        }
        print(f"{key:35s} {np.mean(d):10.4f} {W:10.1f} {p:12.2e}")

    # Session-4 representativeness: where does session_idx=4 sit?
    print("\n=== Session 4 representativeness ===")
    if 4 in common:
        median_per_arch = {n: float(np.median(arrays["pearson_r"][n]))
                           for n in [s[0] for s in SLUGS]}
        sess4_per_arch = {}
        for name in [s[0] for s in SLUGS]:
            v4 = arrays["pearson_r"][name][common.index(4)]
            sess4_per_arch[name] = float(v4)
        # Compute |session_4 r - median r| per arch
        deltas = {n: sess4_per_arch[n] - median_per_arch[n]
                  for n in sess4_per_arch}
        print(f"{'Arch':14s} {'sess4 r':>10s} {'median r':>10s} "
              f"{'delta':>10s}")
        for n in [s[0] for s in SLUGS]:
            print(f"{n:14s} {sess4_per_arch[n]:10.4f} "
                  f"{median_per_arch[n]:10.4f} {deltas[n]:+10.4f}")
        max_abs_delta = max(abs(v) for v in deltas.values())
        summary["session_4"] = {
            "sess4_r": sess4_per_arch,
            "median_r": median_per_arch,
            "delta_from_median": deltas,
            "max_abs_delta": max_abs_delta,
        }
        print(f"\nMax |delta from median| across 7 archs: {max_abs_delta:.4f}")

        # Also: which session is closest to the median across all 7 archs?
        # Use sum of abs deviations from per-arch median
        total_dev = np.zeros(len(common))
        for name in [s[0] for s in SLUGS]:
            arr = arrays["pearson_r"][name]
            med = np.median(arr)
            total_dev += np.abs(arr - med)
        order = np.argsort(total_dev)
        top5 = [(int(common[i]), float(total_dev[i])) for i in order[:5]]
        print(f"\nTop-5 most-representative sessions "
              f"(min total |r - median| across 7 archs):")
        for sidx, td in top5:
            print(f"  session_{sidx:03d}: total dev = {td:.4f}")
        summary["session_4"]["most_representative_sessions"] = top5
    else:
        print("session_idx=4 not in common set!")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
