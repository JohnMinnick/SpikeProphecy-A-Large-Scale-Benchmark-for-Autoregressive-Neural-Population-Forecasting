"""Trivial forecasting baselines (persistence, train-mean).

Reviewer feedback: "Predict the bin-wise mean firing rate" and "predict the
previous bin" are standard sanity floors. Including them calibrates how to
read r=0.5 — is that 5x persistence, or 1.5x?

Two baselines, both per-session:
  Persistence: y_hat(t+1) = y(t)        (last-bin replication)
  Mean:        y_hat(t+1) = mean(y_train, axis=0)  (constant per neuron)

Output structure matches the other baselines (per_session + weighted_avg).
"""

import json
import os
import time
from pathlib import Path

import boto3
import numpy as np

BUCKET = "braingeneersdev"
S3_CACHE_PREFIX = "<anon>/spike-prophecy/inputs/steinmetz-session-cache"
S3_OUTPUT_KEY = "<anon>/spike-prophecy/outputs/glm-trivial-baselines/metrics.json"
LOCAL_CACHE = Path("/data/steinmetz_cache")

HISTORY_BINS = 10
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get(
        "ENDPOINT",
        os.environ.get("S3_ENDPOINT", "https://s3-west.nrp-nautilus.io"),
    ),
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)


def download_s3_cache(prefix, local_dir):
    local_dir.mkdir(parents=True, exist_ok=True)
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, MaxKeys=500)
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        fname = key.split("/")[-1]
        local_path = local_dir / fname
        if not local_path.exists():
            print(f"  Downloading {fname} ({obj['Size']:,} bytes)...")
            s3.download_file(BUCKET, key, str(local_path))
    return local_dir


def compute_pop_metrics(pred, true):
    eps = 1e-8
    pred = pred.astype(np.float64)
    true = true.astype(np.float64)
    n_neurons = pred.shape[1]

    rs = np.zeros(n_neurons)
    for j in range(n_neurons):
        p, t = pred[:, j], true[:, j]
        if np.std(p) < eps or np.std(t) < eps:
            rs[j] = 0.0
        else:
            rs[j] = np.corrcoef(p, t)[0, 1]
    pearson_r = float(np.nanmean(rs))

    pop_pred = pred.sum(axis=1)
    pop_true = true.sum(axis=1)
    pop_rate_r = (
        float(np.corrcoef(pop_pred, pop_true)[0, 1])
        if np.std(pop_pred) > eps and np.std(pop_true) > eps else 0.0
    )

    spatial_rs = []
    for t in range(pred.shape[0]):
        if np.std(pred[t]) > eps and np.std(true[t]) > eps:
            spatial_rs.append(np.corrcoef(pred[t], true[t])[0, 1])
    spatial_r = float(np.nanmean(spatial_rs)) if spatial_rs else 0.0

    norm_p = np.linalg.norm(pred, axis=1)
    norm_t = np.linalg.norm(true, axis=1)
    dot = (pred * true).sum(axis=1)
    den = norm_p * norm_t
    cos_per_t = np.where(den > eps, dot / (den + eps), 0.0)
    cosine_sim = float(np.nanmean(cos_per_t))

    poisson_nll = float(np.mean(np.clip(pred, eps, None)
                                - true * np.log(np.clip(pred, eps, None))))
    mae = float(np.mean(np.abs(pred - true)))

    return {
        "pearson_r": pearson_r, "pop_rate_r": pop_rate_r,
        "spatial_r": spatial_r, "cosine_sim": cosine_sim,
        "poisson_nll": poisson_nll, "mae": mae,
    }


def aggregate(per_session):
    weights = np.array([s["num_neurons"] for s in per_session], dtype=float)
    weights /= weights.sum()
    weighted_avg = {}
    for k in ("pearson_r", "pop_rate_r", "spatial_r", "cosine_sim",
              "poisson_nll", "mae"):
        weighted_avg[k] = float(
            np.sum([w * s[k] for w, s in zip(weights, per_session)])
        )
    sess_se = {}
    for k in ("pearson_r", "pop_rate_r", "spatial_r", "cosine_sim"):
        vals = np.array([s[k] for s in per_session])
        sess_se[k] = {
            "mean": float(vals.mean()),
            "se": float(vals.std(ddof=1) / np.sqrt(len(vals))),
        }
    return weighted_avg, sess_se


def main():
    t_start = time.time()
    print("Trivial forecasting baselines: persistence + train-mean")

    cache_dir = download_s3_cache(S3_CACHE_PREFIX, LOCAL_CACHE)
    metadata = json.load(open(cache_dir / "metadata.json"))
    sessions = metadata.get("sessions", [])

    persistence_sessions = []
    mean_sessions = []

    for sess_idx, sess_info in enumerate(sessions):
        sess_name = sess_info.get("name",
            sess_info.get("session_id", f"session_{sess_idx:03d}"))
        npy_name = sess_info.get("npy", f"session_{sess_idx:03d}.npy")
        try:
            data = np.load(str(cache_dir / npy_name)).astype(np.float32)
        except Exception as e:
            print(f"  skip {sess_name}: {e}")
            continue
        if data.shape[0] < HISTORY_BINS + 50 or data.shape[1] == 0:
            continue

        T_total, M = data.shape
        n = T_total - HISTORY_BINS
        # Build the (X, y) windowed view for split parity with the GLMs
        y_pred_persist = np.zeros((n, M), dtype=np.float32)
        y_true = np.zeros((n, M), dtype=np.float32)
        for i in range(n):
            # persistence: predict y(t+1) = y(t) (last bin of history)
            y_pred_persist[i] = data[i + HISTORY_BINS - 1]
            y_true[i] = data[i + HISTORY_BINS]

        nt = int(TRAIN_FRAC * n)
        nv = int(VAL_FRAC * n)
        slice_val = slice(nt, nt + nv)
        # Mean: predict the per-neuron train-set mean (constant)
        train_mean = y_true[:nt].mean(axis=0)  # (M,)
        y_pred_mean = np.broadcast_to(train_mean, (slice_val.stop - slice_val.start, M))

        m_persist = compute_pop_metrics(y_pred_persist[slice_val], y_true[slice_val])
        m_persist.update({
            "session_idx": sess_idx, "session_id": sess_name,
            "num_neurons": int(M),
        })
        m_mean = compute_pop_metrics(y_pred_mean, y_true[slice_val])
        m_mean.update({
            "session_idx": sess_idx, "session_id": sess_name,
            "num_neurons": int(M),
        })
        persistence_sessions.append(m_persist)
        mean_sessions.append(m_mean)

        print(f"  s{sess_idx:02d} {sess_name[:40]:<40} "
              f"persist r={m_persist['pearson_r']:.3f}  "
              f"mean r={m_mean['pearson_r']:.3f}")

    pers_wt, pers_se = aggregate(persistence_sessions)
    mean_wt, mean_se = aggregate(mean_sessions)
    print()
    print(f"Persistence:  weighted r = {pers_wt['pearson_r']:.4f}  "
          f"(per-session mean = {pers_se['pearson_r']['mean']:.4f} "
          f"+/- {pers_se['pearson_r']['se']:.4f})")
    print(f"Train-mean:   weighted r = {mean_wt['pearson_r']:.4f}  "
          f"(per-session mean = {mean_se['pearson_r']['mean']:.4f} "
          f"+/- {mean_se['pearson_r']['se']:.4f})")

    output = {
        "persistence": {
            "weighted_avg": pers_wt,
            "session_means_se": pers_se,
            "per_session": persistence_sessions,
        },
        "train_mean": {
            "weighted_avg": mean_wt,
            "session_means_se": mean_se,
            "per_session": mean_sessions,
        },
        "n_sessions": len(persistence_sessions),
        "elapsed_s": time.time() - t_start,
    }
    body = json.dumps(output, indent=2).encode("utf-8")
    s3.put_object(Bucket=BUCKET, Key=S3_OUTPUT_KEY, Body=body)
    print(f"\nUploaded results to s3://{BUCKET}/{S3_OUTPUT_KEY}")


if __name__ == "__main__":
    main()
