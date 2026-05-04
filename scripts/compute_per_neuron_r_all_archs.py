"""Compute per-neuron Pearson r and Fano factor for all 7 architectures
across all 39 Steinmetz sessions.

Output (uploaded to S3):
  <anon>/spike-prophecy/outputs/per-neuron-fano-7arch/per_neuron_data.npz
  containing:
    fano_per_session  : list of arrays, one per session, shape (M_i,)
    pn_r[arch]        : list of arrays, one per session, shape (M_i,)
    session_idx       : list of session indices
    arch_names        : the 7 architecture names
"""

import os
import json
import sys
import time
from pathlib import Path

import boto3
import numpy as np
import torch
import yaml

sys.path.insert(0, "/workspace")

# Bootstrap: the v22 NRP image predates HGRN2/GatedDeltaNet; download the
# updated model source files from S3 and overwrite the in-image versions
# before any imports of `src.models`.
def _bootstrap_models():
    import boto3 as _boto3
    bs = _boto3.client(
        "s3",
        endpoint_url=os.environ.get("ENDPOINT", "https://s3-west.nrp-nautilus.io"),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    target_dir = Path("/workspace/src/models")
    target_dir.mkdir(parents=True, exist_ok=True)
    files = [
        ("common.py", "<anon>/spike-prophecy/scripts/_bootstrap/common.py"),
        ("hgrn2_baseline.py",
         "<anon>/spike-prophecy/scripts/_bootstrap/hgrn2_baseline.py"),
        ("gated_delta_baseline.py",
         "<anon>/spike-prophecy/scripts/_bootstrap/gated_delta_baseline.py"),
        ("transformer_baseline.py",
         "<anon>/spike-prophecy/scripts/_bootstrap/transformer_baseline.py"),
    ]
    for fname, key in files:
        try:
            bs.download_file("braingeneersdev", key, str(target_dir / fname))
            print(f"  bootstrapped {fname}")
        except Exception as e:
            print(f"  bootstrap {fname} failed: {e}")


_bootstrap_models()


def _check_cuda():
    """Fail-fast if the assigned GPU is broken — saves us from a 5min
    image-pull only to crash later.  NRP sometimes lands pods on nodes
    whose GPUs are 'busy or unavailable'."""
    import torch as _torch
    if not _torch.cuda.is_available():
        raise SystemExit("CUDA not available on this pod — exit for retry")
    try:
        x = _torch.zeros(8, 8, device="cuda")
        y = x @ x
        y.cpu()
        print(f"  CUDA OK (device={_torch.cuda.get_device_name(0)})")
    except Exception as e:
        raise SystemExit(f"CUDA sanity check failed: {e}")


_check_cuda()


BUCKET = "braingeneersdev"
S3_OUT = "<anon>/spike-prophecy/outputs/per-neuron-fano-7arch/per_neuron_data.npz"
LOCAL = Path("/data/work")
LOCAL.mkdir(parents=True, exist_ok=True)
HISTORY_BINS = 10
M_TRAIN = 1240

ARCHS = [
    # (arch_key, slug, in_image_config_path, s3_filename_fallback, display)
    # ONLY the 3 missing archs — Mamba/Transformer/LRU/SNN per-neuron
    # arrays already exist locally at outputs/eval-suite/<arch>/
    # per_neuron_arrays.npz from the canonical eval run.
    ("hgrn2",       "2026-04-21_2026-04-21_baseline-hgrn2-v1",
     "/workspace/configs/teacher/nrp_teacher_hgrn2.yaml",
     "nrp_teacher_hgrn2.yaml",       "HGRN2"),
    ("gated_delta", "2026-04-22_2026-04-22_baseline-gated-delta-v1",
     "/workspace/configs/teacher/nrp_teacher_gated_delta.yaml",
     "nrp_teacher_gated_delta.yaml", "GatedDelta"),
    ("lstm",        "2026-04-15_baseline-lstm-v23",
     "/workspace/configs/archive/teacher/nrp_teacher.yaml",
     "nrp_teacher_lstm.yaml",        "LSTM"),
]
SNN_SLUG = "2026-04-22_snn-standalone-3l-steinmetz"
SNN_CFG = "/workspace/configs/student/standalone_snn_3l.yaml"

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("ENDPOINT", "https://s3-west.nrp-nautilus.io"),
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)


def fetch(key, dest):
    if dest.exists():
        return dest
    s3.download_file(BUCKET, key, str(dest))
    return dest


def fetch_config(local_path, name_for_s3):
    """Return a path to a usable config file.  Falls back to S3 if the
    in-image path doesn't exist."""
    if Path(local_path).exists():
        return str(local_path)
    s3_key = f"<anon>/spike-prophecy/scripts/{name_for_s3}"
    dest = LOCAL / name_for_s3
    print(f"  config not in image, fetching {s3_key}")
    s3.download_file(BUCKET, s3_key, str(dest))
    return str(dest)


def window(data, T):
    n = data.shape[0] - T
    X = np.zeros((n, T, data.shape[1]), dtype=np.float32)
    y = np.zeros((n, data.shape[1]), dtype=np.float32)
    for i in range(n):
        X[i] = data[i:i + T]
        y[i] = data[i + T]
    return X, y


def _unwrap(pred):
    if isinstance(pred, dict):
        for k in ("rate", "rates", "output", "out"):
            if k in pred:
                return pred[k]
        return list(pred.values())[0]
    if isinstance(pred, (list, tuple)):
        return pred[0]
    return pred


def per_neuron_r(gt, pred):
    """Vectorised per-neuron Pearson r — gt, pred are (T, M).

    ~100x faster than the per-neuron Python loop on M=700 arrays."""
    gt_mean = gt.mean(axis=0, keepdims=True)
    pr_mean = pred.mean(axis=0, keepdims=True)
    gt_c = gt - gt_mean
    pr_c = pred - pr_mean
    num = (gt_c * pr_c).sum(axis=0)
    gt_ss = (gt_c * gt_c).sum(axis=0)
    pr_ss = (pr_c * pr_c).sum(axis=0)
    denom = np.sqrt(gt_ss * pr_ss)
    pn = np.where(denom > 1e-12, num / denom, np.nan).astype(np.float32)
    # Also mask out neurons whose gt or pred is constant (zero std).
    mask = (gt.std(axis=0) > 0) & (pred.std(axis=0) > 0)
    pn = np.where(mask, pn, np.nan)
    return pn.astype(np.float32)


def per_neuron_fano(gt):
    """Fano factor per neuron from GT (T, M)."""
    means = gt.mean(axis=0)
    vars_ = gt.var(axis=0)
    fano = np.full(gt.shape[1], np.nan, dtype=np.float32)
    valid = means > 1e-3
    fano[valid] = vars_[valid] / means[valid]
    return fano


def run_inference(model, X, n_actual, batch=128, device="cuda"):
    model.eval()
    out = np.zeros((X.shape[0], n_actual), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            try:
                pred = model(xb)
            except TypeError:
                mask = torch.ones(xb.shape[0], xb.shape[2], device=device)
                pred = model(xb, mask=mask)
            pred = _unwrap(pred)
            arr = pred.detach().float().cpu().numpy()
            out[i:i + batch] = arr[:, :n_actual]
    return out


def _load_session(sess, idx):
    """Load and window one session, slice to VAL split only.

    Returns (X_val, gt_val, fano_val, n_actual).  Val split is bins
    [train_end, val_end) under the 70/15/15 temporal_split convention
    in src/data/spike_dataset.py — matches the existing eval-suite
    per-neuron r values that the cached 4-arch data was computed on."""
    n_actual = sess.get("num_units")
    n_bins = sess.get("num_bins")
    npy_name = sess.get("npy", f"session_{idx:03d}.npy")
    fetch(f"<anon>/spike-prophecy/inputs/steinmetz-session-cache/{npy_name}",
          LOCAL / npy_name)
    raw = np.load(str(LOCAL / npy_name)).astype(np.float32)
    if raw.shape[0] == n_actual and raw.shape[1] == n_bins:
        raw = raw.T
    raw_padded = np.zeros((raw.shape[0], M_TRAIN), dtype=np.float32)
    raw_padded[:, :n_actual] = raw[:, :n_actual]

    # Compute val window range *before* windowing.  Targets are at index
    # i + HISTORY_BINS for window i.  Val targets fall in
    # [train_end, val_end) = [0.70*T, 0.85*T).  Window starts are
    # therefore [train_end - HISTORY_BINS, val_end - HISTORY_BINS).
    T_total = raw_padded.shape[0]
    train_end = int(T_total * 0.70)
    val_end = int(T_total * 0.85)
    win_start = max(0, train_end - HISTORY_BINS)
    win_end_ex = max(win_start, val_end - HISTORY_BINS)

    X, y = window(raw_padded, HISTORY_BINS)
    X = X[win_start:win_end_ex]
    y = y[win_start:win_end_ex]
    gt_actual = y[:, :n_actual]
    # Fano factor stays computed on the val ground-truth bins (matches
    # what `eval-suite` per-neuron pipeline does).
    fano = per_neuron_fano(gt_actual)
    return X, gt_actual, fano, n_actual


def main():
    # Fetch session metadata
    fetch("<anon>/spike-prophecy/inputs/steinmetz-session-cache/metadata.json",
          LOCAL / "metadata.json")
    md = json.load(open(LOCAL / "metadata.json"))
    sessions = md["sessions"]
    n_sessions = len(sessions)
    print(f"Found {n_sessions} sessions")

    # Streaming approach: outer loop over architectures, inner loop over
    # sessions.  Memory stays bounded — only one session's windowed
    # array lives at a time.
    results = {
        "arch_names": [],
        "session_idx": list(range(n_sessions)),
        "fano_per_session": [None] * n_sessions,
        "pn_r": {},
    }

    from src.models.common import create_teacher_model
    from src.models.student import StudentSNN

    # SNN excluded — per-neuron arrays already exist locally for SNN.
    arch_specs = [
        (k, slug, cfg, s3name, name, "teacher")
        for k, slug, cfg, s3name, name in ARCHS
    ]

    for arch_key, slug, cfg_path, s3_cfg_name, name, kind in arch_specs:
        print(f"\n=== {name} ({arch_key}) ===")
        results["arch_names"].append(name)
        results["pn_r"][name] = [None] * n_sessions

        ckpt_dest = LOCAL / f"ckpt_{arch_key}.pt"
        try:
            fetch(f"<anon>/spike-prophecy/outputs/{slug}/best_model.pt",
                  ckpt_dest)
        except Exception as e:
            print(f"  fetch failed: {e}")
            continue

        try:
            cfg_path = fetch_config(cfg_path, s3_cfg_name)
        except Exception as e:
            print(f"  config fetch failed: {e}, skip")
            continue

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"  yaml load failed: {e}, skip")
            continue
        if kind == "teacher":
            config.setdefault("model", {})["architecture"] = arch_key

        try:
            ckpt = torch.load(str(ckpt_dest), map_location="cpu",
                              weights_only=False)
            state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
            if kind == "teacher":
                model = create_teacher_model(config, M_TRAIN, session_dims=None)
            else:
                model = StudentSNN.from_config(config, M_TRAIN)
            miss, unexp = model.load_state_dict(state, strict=False)
            if miss:
                print(f"  WARN missing: {len(miss)} (e.g. {miss[:2]})")
            model = model.cuda()
            t0 = time.time()
            for i, sess in enumerate(sessions):
                try:
                    X, gt, fano, n_actual = _load_session(sess, i)
                except Exception as e:
                    print(f"  session {i} load failed: {e}")
                    continue
                # Stash Fano on first arch encountered (same for all)
                if results["fano_per_session"][i] is None:
                    results["fano_per_session"][i] = fano
                preds = run_inference(model, X, n_actual)
                pn = per_neuron_r(gt, preds)
                results["pn_r"][name][i] = pn
                # Free memory immediately
                del X, gt, preds
                if (i + 1) % 10 == 0:
                    print(f"    session {i+1}/{n_sessions} done")
            elapsed = time.time() - t0
            print(f"  {name}: {elapsed:.1f}s for {n_sessions} sessions")
            del model
            torch.cuda.empty_cache()
            # Persist after every arch so later failures don't lose work.
            try:
                out_path = LOCAL / "per_neuron_data.npz"

                def _to_obj_arr(lst):
                    return np.array([
                        x if x is not None else np.zeros(0, dtype=np.float32)
                        for x in lst
                    ], dtype=object)
                payload = {
                    "session_idx": np.array(results["session_idx"]),
                    "arch_names": np.array(results["arch_names"]),
                    "fano_per_session": _to_obj_arr(results["fano_per_session"]),
                }
                for nm in results["arch_names"]:
                    payload[f"pn_r__{nm}"] = _to_obj_arr(results["pn_r"][nm])
                np.savez_compressed(str(out_path), **payload)
                s3.upload_file(str(out_path), BUCKET, S3_OUT)
                print(f"  partial saved through {name}")
            except Exception as ee:
                print(f"  partial save failed: {ee}")
        except Exception as e:
            print(f"  {name}: FAILED ({type(e).__name__}: {e})")
            import traceback
            traceback.print_exc()

    # Save as npz with object arrays (variable per-session lengths).
    # Replace None entries with empty arrays so np can serialize.
    print("\n=== Saving ===")
    out_path = LOCAL / "per_neuron_data.npz"

    def _to_obj_arr(lst):
        return np.array([
            x if x is not None else np.zeros(0, dtype=np.float32)
            for x in lst
        ], dtype=object)

    payload = {
        "session_idx": np.array(results["session_idx"]),
        "arch_names": np.array(results["arch_names"]),
        "fano_per_session": _to_obj_arr(results["fano_per_session"]),
    }
    for name in results["arch_names"]:
        payload[f"pn_r__{name}"] = _to_obj_arr(results["pn_r"][name])
    np.savez_compressed(str(out_path), **payload)
    print(f"  Saved: {out_path} ({out_path.stat().st_size:,} bytes)")
    s3.upload_file(str(out_path), BUCKET, S3_OUT)
    print(f"  Uploaded: s3://{BUCKET}/{S3_OUT}")


if __name__ == "__main__":
    main()
