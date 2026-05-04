"""Per-neuron autoregressive Poisson GLM baseline (NRP, self-contained).

Unlike the population GLM in ``src/eval/glm_baseline.py`` (which flattens the
full (T, M) history window into T*M features per neuron and overfits
catastrophically), this version uses ONLY neuron i's own T-step history as
features for neuron i's GLM --- a true no-cross-neuron-information baseline.

Target: SpikeProphecy NeurIPS E&D Table 1 (Steinmetz 39 sessions, 27,212 neurons).

Architecture (per neuron i, per session s):
    log(lambda_i(t+1)) = w_i^T . x_i[t-T+1:t+1] + b_i
    features: T scalars (only neuron i's own T-step history)

Outputs (matches other baselines in Table 1): weighted Pearson r (neuron-count
weighted), pop_rate_r, spatial_r, cosine_sim, Poisson NLL, MAE --- aggregated
across all 39 sessions on the val split.

Usage (NRP job YAML downloads this from S3 and runs it as `python /tmp/...`):
    python scripts/train_glm_autoreg.py
"""

import json
import os
import sys
import time
from pathlib import Path

import boto3
import numpy as np
from sklearn.linear_model import PoissonRegressor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BUCKET = "braingeneersdev"
S3_CACHE_PREFIX = "<anon>/spike-prophecy/inputs/steinmetz-session-cache"
S3_OUTPUT_KEY = "<anon>/spike-prophecy/outputs/glm-autoreg-baseline/metrics.json"
LOCAL_CACHE = Path("/data/steinmetz_cache")

HISTORY_BINS = 10           # T --- matches the deep baselines
GLM_ALPHA = 1e-4            # L2 regularization strength
GLM_MAX_ITER = 300
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15             # test = remaining 15%

# ---------------------------------------------------------------------------
# S3 client
# ---------------------------------------------------------------------------
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get(
        "ENDPOINT",
        os.environ.get("S3_ENDPOINT", "https://s3-west.nrp-nautilus.io"),
    ),
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)


def download_s3_cache(s3_prefix: str, local_dir: Path) -> Path:
    """Download all files under an S3 prefix to a local dir."""
    local_dir.mkdir(parents=True, exist_ok=True)
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=s3_prefix, MaxKeys=500)
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        fname = key.split("/")[-1]
        local_path = local_dir / fname
        if not local_path.exists():
            print(f"  Downloading {fname} ({obj['Size']:,} bytes)...")
            s3.download_file(BUCKET, key, str(local_path))
    print(
        f"  Cache ready: {local_dir} "
        f"({len(list(local_dir.iterdir()))} files)"
    )
    return local_dir


def window_session(data: np.ndarray, history_bins: int):
    """Create (X, y) windowed samples from a (T_total, N) spike-count matrix."""
    T_total, n_neurons = data.shape
    n_samples = T_total - history_bins
    if n_samples <= 0:
        return None
    X = np.zeros((n_samples, history_bins, n_neurons), dtype=np.float32)
    y = np.zeros((n_samples, n_neurons), dtype=np.float32)
    for i in range(n_samples):
        X[i] = data[i:i + history_bins]
        y[i] = data[i + history_bins]
    return X, y


def split_samples(X: np.ndarray, y: np.ndarray):
    """Temporal 70/15/15 train/val/test split (no shuffle)."""
    n = X.shape[0]
    n_train = int(TRAIN_FRAC * n)
    n_val = int(VAL_FRAC * n)
    return {
        "train": (X[:n_train], y[:n_train]),
        "val": (X[n_train:n_train + n_val], y[n_train:n_train + n_val]),
        "test": (X[n_train + n_val:], y[n_train + n_val:]),
    }


def fit_autoreg_glm_per_neuron(X_train: np.ndarray, y_train: np.ndarray):
    """Fit one Poisson GLM per neuron using ONLY that neuron's own T-step history.

    Args:
        X_train: (N_samples, T, M) history windows.
        y_train: (N_samples, M) next-bin spike counts.

    Returns:
        List of M fitted PoissonRegressor models (None for zero-activity channels).
    """
    n_neurons = y_train.shape[1]
    models = []
    for i in range(n_neurons):
        if y_train[:, i].sum() == 0:
            models.append(None)
            continue
        # Feature vector for neuron i: its own T past bins (T features)
        x_i = X_train[:, :, i]
        glm = PoissonRegressor(
            alpha=GLM_ALPHA, max_iter=GLM_MAX_ITER, fit_intercept=True,
        )
        glm.fit(x_i, y_train[:, i])
        models.append(glm)
    return models


def predict_autoreg(models, X: np.ndarray) -> np.ndarray:
    """Predict rates using autoreg GLMs. Shape (N_samples, M)."""
    n_samples = X.shape[0]
    n_neurons = len(models)
    preds = np.zeros((n_samples, n_neurons), dtype=np.float64)
    for i, model in enumerate(models):
        if model is not None:
            preds[:, i] = model.predict(X[:, :, i])
    return preds


def compute_pop_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Compute the paper's population-metric decomposition for one session.

    Args:
        pred: (N_samples, M) predicted rates.
        true: (N_samples, M) ground-truth spike counts.

    Returns:
        dict with pearson_r (per-neuron mean), pop_rate_r, spatial_r,
        cosine_sim, poisson_nll, mae.
    """
    eps = 1e-8
    pred = pred.astype(np.float64)
    true = true.astype(np.float64)

    # Per-neuron Pearson r (match training/archive definition: mean over neurons)
    n_neurons = pred.shape[1]
    rs = np.zeros(n_neurons)
    for j in range(n_neurons):
        p, t = pred[:, j], true[:, j]
        if np.std(p) < eps or np.std(t) < eps:
            rs[j] = 0.0
        else:
            rs[j] = np.corrcoef(p, t)[0, 1]
    pearson_r = float(np.nanmean(rs))

    # Population rate r: sum over neurons, then correlate across time
    pop_pred = pred.sum(axis=1)
    pop_true = true.sum(axis=1)
    if np.std(pop_pred) > eps and np.std(pop_true) > eps:
        pop_rate_r = float(np.corrcoef(pop_pred, pop_true)[0, 1])
    else:
        pop_rate_r = 0.0

    # Spatial r: per-timebin cross-neuron correlation, averaged over time
    spatial_rs = []
    for t in range(pred.shape[0]):
        p_t, t_t = pred[t], true[t]
        if np.std(p_t) > eps and np.std(t_t) > eps:
            spatial_rs.append(np.corrcoef(p_t, t_t)[0, 1])
    spatial_r = float(np.nanmean(spatial_rs)) if spatial_rs else 0.0

    # Cosine sim: per-timebin, averaged
    norm_p = np.linalg.norm(pred, axis=1)
    norm_t = np.linalg.norm(true, axis=1)
    dot = (pred * true).sum(axis=1)
    den = norm_p * norm_t
    cos_per_t = np.where(den > eps, dot / (den + eps), 0.0)
    cosine_sim = float(np.nanmean(cos_per_t))

    # Poisson NLL (lambda - y*log(lambda+eps))
    poisson_nll = float(np.mean(pred - true * np.log(pred + eps)))

    # MAE
    mae = float(np.mean(np.abs(pred - true)))

    return {
        "pearson_r": pearson_r,
        "pop_rate_r": pop_rate_r,
        "spatial_r": spatial_r,
        "cosine_sim": cosine_sim,
        "poisson_nll": poisson_nll,
        "mae": mae,
    }


def main():
    t_start = time.time()
    print("=" * 70)
    print("  Per-neuron AUTOREGRESSIVE Poisson GLM baseline")
    print("  (features: neuron i's own T=10 history bins only)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Download session cache
    # ------------------------------------------------------------------
    cache_dir = download_s3_cache(S3_CACHE_PREFIX, LOCAL_CACHE)
    metadata_path = cache_dir / "metadata.json"
    metadata = json.load(open(metadata_path))
    sessions = metadata.get("sessions", [])
    m_max = metadata.get("m_max")
    print(f"  m_max={m_max}, n_sessions={len(sessions)}")

    # ------------------------------------------------------------------
    # 2. Per-session fit + eval
    # ------------------------------------------------------------------
    per_session = []
    for sess_idx, sess_info in enumerate(sessions):
        n_neurons = sess_info.get(
            "num_units", sess_info.get("n_units", m_max),
        )
        session_name = sess_info.get(
            "name", sess_info.get("session_id", f"session_{sess_idx:03d}"),
        )
        print(
            f"\n--- Session {sess_idx + 1}/{len(sessions)}: "
            f"{session_name} ({n_neurons} neurons) ---"
        )
        sess_t = time.time()

        # Load the padded (T_total, M_max) matrix, trim to real neurons
        sess_file = cache_dir / f"session_{sess_idx:03d}.npy"
        if not sess_file.exists():
            # Fallback for alternative naming
            candidates = sorted(cache_dir.glob(f"*{sess_idx:03d}*.npy"))
            if not candidates:
                print(f"  WARN: no .npy for session {sess_idx}, skipping")
                continue
            sess_file = candidates[0]
        data = np.load(sess_file)[:, :n_neurons]

        windowed = window_session(data, HISTORY_BINS)
        if windowed is None:
            print(f"  WARN: session too short, skipping")
            continue
        X, y = windowed
        splits = split_samples(X, y)
        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        print(
            f"  train={X_train.shape[0]}, val={X_val.shape[0]}, "
            f"features=T={HISTORY_BINS} per neuron"
        )

        # Fit per-neuron autoreg GLMs on train
        models = fit_autoreg_glm_per_neuron(X_train, y_train)
        n_fitted = sum(1 for m in models if m is not None)

        # Evaluate on val
        preds_val = predict_autoreg(models, X_val)
        sess_metrics = compute_pop_metrics(preds_val, y_val)
        sess_time = time.time() - sess_t
        print(
            f"  val r={sess_metrics['pearson_r']:+.4f} | "
            f"pop_rate_r={sess_metrics['pop_rate_r']:+.4f} | "
            f"spatial_r={sess_metrics['spatial_r']:+.4f} | "
            f"cos={sess_metrics['cosine_sim']:.4f} | "
            f"fit {sess_time:.1f}s ({n_fitted}/{n_neurons} neurons)"
        )

        per_session.append({
            "session_idx": sess_idx,
            "session_id": session_name,
            "num_neurons": n_neurons,
            "n_fitted": n_fitted,
            "fit_time_s": round(sess_time, 2),
            **sess_metrics,
        })

        # Free memory between sessions
        del models, preds_val, X_train, y_train, X_val, y_val, X, y, data

    # ------------------------------------------------------------------
    # 3. Aggregate metrics (neuron-count weighted)
    # ------------------------------------------------------------------
    total_n = sum(s["num_neurons"] for s in per_session)
    def wmean(key):
        if total_n == 0:
            return 0.0
        return sum(s[key] * s["num_neurons"] for s in per_session) / total_n

    agg = {
        "pearson_r": round(wmean("pearson_r"), 4),
        "pop_rate_r": round(wmean("pop_rate_r"), 4),
        "spatial_r": round(wmean("spatial_r"), 4),
        "cosine_sim": round(wmean("cosine_sim"), 4),
        "poisson_nll": round(wmean("poisson_nll"), 4),
        "mae": round(wmean("mae"), 4),
    }

    result = {
        "model_type": "poisson_glm_autoreg",
        "description": (
            "Per-neuron autoregressive Poisson GLM: each neuron's model uses "
            "only its own T=10 history bins as features (no cross-neuron "
            "information). Contrasts with the population GLM which flattens "
            "the full (T, M) history and overfits catastrophically."
        ),
        "dataset": "steinmetz-39-session",
        "history_bins": HISTORY_BINS,
        "glm_alpha": GLM_ALPHA,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "n_sessions": len(per_session),
        "total_neurons": total_n,
        "weighted_avg": agg,
        "per_session": per_session,
        "total_time_s": round(time.time() - t_start, 1),
    }

    # ------------------------------------------------------------------
    # 4. Print summary and upload
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  AGGREGATE RESULTS (neuron-count weighted, val split)")
    print("=" * 70)
    print(f"  sessions    : {len(per_session)}")
    print(f"  total N     : {total_n}")
    print(f"  pearson_r   : {agg['pearson_r']:+.4f}")
    print(f"  pop_rate_r  : {agg['pop_rate_r']:+.4f}")
    print(f"  spatial_r   : {agg['spatial_r']:+.4f}")
    print(f"  cosine_sim  : {agg['cosine_sim']:.4f}")
    print(f"  poisson_nll : {agg['poisson_nll']:.4f}")
    print(f"  mae         : {agg['mae']:.4f}")
    print(f"  total time  : {result['total_time_s']:.1f}s")

    out_bytes = json.dumps(result, indent=2).encode()
    s3.put_object(Bucket=BUCKET, Key=S3_OUTPUT_KEY, Body=out_bytes)
    print(f"\nUploaded to s3://{BUCKET}/{S3_OUTPUT_KEY}")


if __name__ == "__main__":
    main()
