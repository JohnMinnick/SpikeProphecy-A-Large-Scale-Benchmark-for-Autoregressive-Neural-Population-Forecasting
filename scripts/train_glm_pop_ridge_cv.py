"""Population GLM with cross-validated ridge regularization (NRP, self-contained).

Addresses NeurIPS reviewer concern: the existing population GLM uses a fixed
small alpha (10^-4) on T*M ~= 7000 features per neuron, which is guaranteed
to overfit on temporally-split data — the reported r = -0.015 is the
canonical leakage-suite catch, not a fair linear baseline.

This script tunes alpha on the val split via a coarse grid, picks the best
per-session alpha, then reports the standard pop_metrics on val.

Output: jrm/spike-prophecy/outputs/glm-pop-ridge-cv/metrics.json
"""

import json
import os
import time
from pathlib import Path

import boto3
import numpy as np
from sklearn.linear_model import Ridge

BUCKET = "braingeneersdev"
S3_CACHE_PREFIX = "jrm/spike-prophecy/inputs/steinmetz-session-cache"
S3_OUTPUT_KEY = "jrm/spike-prophecy/outputs/glm-pop-ridge-cv/metrics.json"
LOCAL_CACHE = Path("/data/steinmetz_cache")

HISTORY_BINS = 10
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
ALPHA_GRID = [10.0, 100.0, 1000.0, 10000.0]

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


def window_session(data, history_bins):
    T_total, n_neurons = data.shape
    n_samples = T_total - history_bins
    if n_samples <= 0:
        return None
    X = np.zeros((n_samples, history_bins * n_neurons), dtype=np.float32)
    y = np.zeros((n_samples, n_neurons), dtype=np.float32)
    for i in range(n_samples):
        # Flatten the full (T, M) history into one feature vector
        X[i] = data[i:i + history_bins].reshape(-1)
        y[i] = data[i + history_bins]
    return X, y


def split(X, y):
    n = X.shape[0]
    nt = int(TRAIN_FRAC * n)
    nv = int(VAL_FRAC * n)
    return {
        "train": (X[:nt], y[:nt]),
        "val":   (X[nt:nt + nv], y[nt:nt + nv]),
        "test":  (X[nt + nv:], y[nt + nv:]),
    }


def fit_pop_ridge_cv(X_train, y_train, X_val, y_val):
    """Fit a single multi-output Ridge with alpha tuned by val r.

    The features are shared across neurons (full population history),
    so we fit one multi-output Ridge per session rather than one per
    neuron — much faster, same expressive power.
    """
    best_alpha = None
    best_score = -np.inf
    best_model = None

    for alpha in ALPHA_GRID:
        model = Ridge(alpha=alpha, fit_intercept=True, copy_X=False)
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
        # Softplus to keep predictions non-negative (rate-like)
        pred_val = np.log1p(np.exp(np.clip(pred_val, -50, 50)))
        # Score: mean per-neuron Pearson r on val
        r_per_neuron = []
        for j in range(y_val.shape[1]):
            if np.std(y_val[:, j]) < 1e-8 or np.std(pred_val[:, j]) < 1e-8:
                continue
            r_per_neuron.append(np.corrcoef(pred_val[:, j], y_val[:, j])[0, 1])
        score = float(np.nanmean(r_per_neuron)) if r_per_neuron else 0.0
        print(f"    alpha={alpha:>10.1f}  val mean r = {score:.4f}")
        if score > best_score:
            best_score = score
            best_alpha = alpha
            best_model = model

    return best_model, best_alpha, best_score


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


def main():
    t_start = time.time()
    print("=" * 70)
    print("  Population GLM with CV-tuned ridge regularization")
    print(f"  Alpha grid: {ALPHA_GRID}")
    print("=" * 70)

    cache_dir = download_s3_cache(S3_CACHE_PREFIX, LOCAL_CACHE)
    metadata = json.load(open(cache_dir / "metadata.json"))
    sessions = metadata.get("sessions", [])

    per_session = []
    for sess_idx, sess_info in enumerate(sessions):
        sess_name = sess_info.get(
            "name", sess_info.get("session_id", f"session_{sess_idx:03d}"),
        )
        n_neurons = sess_info.get("num_units", sess_info.get("n_units", 0))
        print(f"\n--- Session {sess_idx + 1}/{len(sessions)}: "
              f"{sess_name} ({n_neurons} neurons) ---")
        sess_t = time.time()

        # Load this session's spike-count matrix; sessions are zero-padded
        # to M_max across the corpus, so truncate to actual neurons.
        npy_name = sess_info.get("npy", f"session_{sess_idx:03d}.npy")
        data_full = np.load(str(cache_dir / npy_name)).astype(np.float32)
        actual_n = sess_info.get("num_units", sess_info.get("n_units", 0))
        if actual_n == 0 or data_full.shape[1] == 0:
            print("  (empty session, skipping)")
            continue
        data = data_full[:, :actual_n]
        print(f"  Truncated padded ({data_full.shape[1]} cols) -> "
              f"actual ({actual_n} neurons)")

        windowed = window_session(data, HISTORY_BINS)
        if windowed is None:
            print("  (too short, skipping)")
            continue
        X, y = windowed
        splits = split(X, y)
        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]

        if X_train.shape[0] < 50:
            print(f"  (only {X_train.shape[0]} train samples, skipping)")
            continue

        print(f"  Features per sample: {X_train.shape[1]} "
              f"({HISTORY_BINS} bins x {y_train.shape[1]} neurons)")
        print(f"  Train/val samples: {X_train.shape[0]}/{X_val.shape[0]}")

        model, best_alpha, val_mean_r = fit_pop_ridge_cv(
            X_train, y_train, X_val, y_val,
        )

        # Final eval: use best_alpha model on val, full pop_metrics
        pred_val = model.predict(X_val)
        pred_val = np.log1p(np.exp(np.clip(pred_val, -50, 50)))
        metrics = compute_pop_metrics(pred_val, y_val)
        metrics.update({
            "session_idx": sess_idx,
            "session_id": sess_name,
            "num_neurons": int(y_train.shape[1]),
            "best_alpha": float(best_alpha),
            "val_mean_r_at_best": float(val_mean_r),
            "n_train": int(X_train.shape[0]),
            "n_val": int(X_val.shape[0]),
        })
        per_session.append(metrics)
        print(f"  Best alpha = {best_alpha}, val pearson r = {metrics['pearson_r']:.4f}, "
              f"pop_rate_r = {metrics['pop_rate_r']:.4f}, "
              f"spatial_r = {metrics['spatial_r']:.4f}, "
              f"cosine = {metrics['cosine_sim']:.4f}")
        print(f"  ({time.time() - sess_t:.1f}s)")

    # Aggregate (activity-weighted by neuron count)
    weights = np.array([s["num_neurons"] for s in per_session], dtype=float)
    weights /= weights.sum()
    weighted_avg = {}
    for k in ("pearson_r", "pop_rate_r", "spatial_r", "cosine_sim",
              "poisson_nll", "mae"):
        weighted_avg[k] = float(
            np.sum([w * s[k] for w, s in zip(weights, per_session)])
        )

    # Also compute simple mean and SE across sessions
    metric_means_se = {}
    for k in ("pearson_r", "pop_rate_r", "spatial_r", "cosine_sim"):
        vals = np.array([s[k] for s in per_session])
        metric_means_se[k] = {
            "mean": float(vals.mean()),
            "se": float(vals.std(ddof=1) / np.sqrt(len(vals))),
        }

    output = {
        "model": "Population GLM (CV-tuned Ridge)",
        "alpha_grid": ALPHA_GRID,
        "n_sessions": len(per_session),
        "weighted_avg": weighted_avg,
        "session_means_se": metric_means_se,
        "per_session": per_session,
        "elapsed_s": time.time() - t_start,
    }

    print("\n" + "=" * 70)
    print(f"  Final weighted Pearson r = {weighted_avg['pearson_r']:.4f}")
    print(f"  Mean per-session r = {metric_means_se['pearson_r']['mean']:.4f} "
          f"+/- {metric_means_se['pearson_r']['se']:.4f}")
    print(f"  Total time: {output['elapsed_s']:.1f}s")
    print("=" * 70)

    # Upload to S3
    body = json.dumps(output, indent=2).encode("utf-8")
    s3.put_object(Bucket=BUCKET, Key=S3_OUTPUT_KEY, Body=body)
    print(f"\nUploaded results to s3://{BUCKET}/{S3_OUTPUT_KEY}")


if __name__ == "__main__":
    main()
