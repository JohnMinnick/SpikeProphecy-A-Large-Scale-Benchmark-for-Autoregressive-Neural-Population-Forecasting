"""Autoregressive rollout for all 7 architectures on Steinmetz session 4.

For each architecture, computes K-step autoregressive predictions
(K=1..20) by iteratively feeding the model's own predictions back as
input history. Reports pop-vector r and per-neuron r at each K.

Output (uploaded to S3):
  <anon>/spike-prophecy/outputs/ar-rollout-7arch/ar_rollout.json
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


def _bootstrap_models():
    """v22 NRP image predates HGRN2/GatedDeltaNet — patch in the new
    model files from S3 before any src.models imports."""
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

BUCKET = "braingeneersdev"
S3_OUT = "<anon>/spike-prophecy/outputs/ar-rollout-7arch/ar_rollout.json"
LOCAL = Path("/data/work")
LOCAL.mkdir(parents=True, exist_ok=True)
SESSION_IDX = 4
HISTORY_BINS = 10
M_TRAIN = 1240
K_STEPS = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]

ARCHS = [
    ("mamba",       "2026-03-26_baseline-mamba-v12",
     "/workspace/configs/teacher/nrp_teacher_mamba.yaml",
     "nrp_teacher_mamba.yaml",       "Mamba"),
    ("hgrn2",       "2026-04-21_2026-04-21_baseline-hgrn2-v1",
     "/workspace/configs/teacher/nrp_teacher_hgrn2.yaml",
     "nrp_teacher_hgrn2.yaml",       "HGRN2"),
    ("transformer", "2026-03-25_baseline-transformer-v12",
     "/workspace/configs/archive/teacher/nrp_teacher_transformer.yaml",
     "nrp_teacher_transformer.yaml", "Transformer"),
    ("gated_delta", "2026-04-22_2026-04-22_baseline-gated-delta-v1",
     "/workspace/configs/teacher/nrp_teacher_gated_delta.yaml",
     "nrp_teacher_gated_delta.yaml", "GatedDelta"),
    ("lru",         "2026-03-25_baseline-lru-v12",
     "/workspace/configs/teacher/nrp_teacher_lru_v2.yaml",
     "nrp_teacher_lru_v2.yaml",      "LRU"),
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


def fetch_config(local_path, s3_filename):
    """Return a path to a usable config — fall back to S3 if missing."""
    if Path(local_path).exists():
        return str(local_path)
    s3_key = f"<anon>/spike-prophecy/scripts/{s3_filename}"
    dest = LOCAL / s3_filename
    print(f"  config not in image, fetching {s3_key}")
    s3.download_file(BUCKET, s3_key, str(dest))
    return str(dest)


def _unwrap(pred):
    if isinstance(pred, dict):
        for k in ("rate", "rates", "output", "out"):
            if k in pred:
                return pred[k]
        return list(pred.values())[0]
    if isinstance(pred, (list, tuple)):
        return pred[0]
    return pred


def per_neuron_r_vec(gt, pred):
    pn = []
    for j in range(gt.shape[1]):
        gj = gt[:, j]
        pj = pred[:, j]
        if gj.std() > 0 and pj.std() > 0:
            gm = gj - gj.mean()
            pm = pj - pj.mean()
            denom = np.sqrt((gm * gm).sum() * (pm * pm).sum())
            if denom > 0:
                pn.append(float((gm * pm).sum() / denom))
    return np.array(pn) if pn else np.array([0.0])


def pop_vector_r(gt, pred):
    """Pearson r between summed-over-neurons time series."""
    g = gt.sum(axis=1)
    p = pred.sum(axis=1)
    if g.std() == 0 or p.std() == 0:
        return 0.0
    gm = g - g.mean()
    pm = p - p.mean()
    return float((gm * pm).sum()
                 / np.sqrt((gm * gm).sum() * (pm * pm).sum()))


def rollout(model, raw, n_actual, k_max=max(K_STEPS), batch=64,
            device="cuda"):
    """K-step autoregressive rollout starting at every valid time index.

    Returns:
      gt_at_k : dict {K: array (n_starts, n_actual)}  ground-truth at t+K
      pred_at_k : dict {K: array (n_starts, n_actual)}  prediction at t+K
    """
    T_total = raw.shape[0]
    n_starts = T_total - HISTORY_BINS - k_max
    if n_starts <= 0:
        raise ValueError(f"Not enough timesteps: T={T_total}")

    # Pre-allocate output buffers
    pred_at_k = {k: np.zeros((n_starts, n_actual), dtype=np.float32)
                 for k in K_STEPS}
    gt_at_k = {k: np.zeros((n_starts, n_actual), dtype=np.float32)
               for k in K_STEPS}

    model.eval()
    with torch.no_grad():
        for s0 in range(0, n_starts, batch):
            s1 = min(s0 + batch, n_starts)
            B = s1 - s0
            # Initial history: shape (B, T, M_TRAIN)
            hist = np.zeros((B, HISTORY_BINS, M_TRAIN), dtype=np.float32)
            for i, s in enumerate(range(s0, s1)):
                hist[i] = raw[s:s + HISTORY_BINS]
            hist_t = torch.from_numpy(hist).to(device)

            for k in range(1, k_max + 1):
                try:
                    pred = model(hist_t)
                except TypeError:
                    mask = torch.ones(B, M_TRAIN, device=device)
                    pred = model(hist_t, mask=mask)
                pred = _unwrap(pred)
                # pred: (B, M_TRAIN)
                pred_np = pred.detach().float().cpu().numpy()
                if k in K_STEPS:
                    pred_at_k[k][s0:s1] = pred_np[:, :n_actual]
                    # ground-truth at step k
                    gt_idx_start = s0 + HISTORY_BINS + k - 1
                    for i, s in enumerate(range(s0, s1)):
                        gt_at_k[k][i + (s0 - s0) if False else 0]  # placeholder
                    # cleaner: vectorize
                    ranges = np.arange(s0, s1) + HISTORY_BINS + k - 1
                    gt_at_k[k][s0:s1] = raw[ranges, :n_actual]
                # Slide history forward: drop earliest, append prediction
                # hist_t shape: (B, T, M_TRAIN); pred shape: (B, M_TRAIN)
                hist_t = torch.cat([hist_t[:, 1:, :],
                                    pred.unsqueeze(1)], dim=1)
    return gt_at_k, pred_at_k


def main():
    fetch("<anon>/spike-prophecy/inputs/steinmetz-session-cache/metadata.json",
          LOCAL / "metadata.json")
    md = json.load(open(LOCAL / "metadata.json"))
    sess = md["sessions"][SESSION_IDX]
    n_actual = sess.get("num_units")
    n_bins = sess.get("num_bins")
    npy_name = sess.get("npy", f"session_{SESSION_IDX:03d}.npy")
    fetch(f"<anon>/spike-prophecy/inputs/steinmetz-session-cache/{npy_name}",
          LOCAL / npy_name)
    raw = np.load(str(LOCAL / npy_name)).astype(np.float32)
    if raw.shape[0] == n_actual and raw.shape[1] == n_bins:
        raw = raw.T
    raw_padded = np.zeros((raw.shape[0], M_TRAIN), dtype=np.float32)
    raw_padded[:, :n_actual] = raw[:, :n_actual]
    print(f"Session {SESSION_IDX}: T_total={raw.shape[0]}, "
          f"n_actual={n_actual}")

    results = {"k_steps": K_STEPS, "archs": {}}

    from src.models.common import create_teacher_model
    from src.models.student import StudentSNN

    def _process_arch(name, model):
        model = model.cuda()
        t0 = time.time()
        gt_k, pred_k = rollout(model, raw_padded, n_actual)
        elapsed = time.time() - t0
        pop_r = []
        neuron_r = []
        for k in K_STEPS:
            pop_r.append(pop_vector_r(gt_k[k], pred_k[k]))
            neuron_r.append(float(np.mean(per_neuron_r_vec(gt_k[k], pred_k[k]))))
        print(f"  {name}: {elapsed:.1f}s, "
              f"pop_r at K=1: {pop_r[0]:.3f}, K=20: {pop_r[-1]:.3f}; "
              f"neuron_r at K=1: {neuron_r[0]:.3f}, K=20: {neuron_r[-1]:.3f}")
        results["archs"][name] = {"pop_r": pop_r, "neuron_r": neuron_r}
        del model
        torch.cuda.empty_cache()
        # Persist after every arch so a later failure doesn't lose work.
        out_path = LOCAL / "ar_rollout.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        try:
            s3.upload_file(str(out_path), BUCKET, S3_OUT)
        except Exception as e:
            print(f"  partial upload failed: {e}")

    for arch_key, slug, cfg_path, s3_cfg_name, name in ARCHS:
        print(f"\n=== {name} ({arch_key}) ===")
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
        config.setdefault("model", {})["architecture"] = arch_key
        try:
            ckpt = torch.load(str(ckpt_dest), map_location="cpu",
                              weights_only=False)
            state = ckpt["model_state_dict"]
            model = create_teacher_model(config, M_TRAIN, session_dims=None)
            miss, unexp = model.load_state_dict(state, strict=False)
            if miss:
                print(f"  WARN missing: {len(miss)}")
            _process_arch(name, model)
        except Exception as e:
            print(f"  {name}: FAILED ({type(e).__name__}: {e})")

    # SNN
    print("\n=== SNN ===")
    snn_ckpt = LOCAL / "ckpt_snn.pt"
    try:
        fetch(f"<anon>/spike-prophecy/outputs/{SNN_SLUG}/best_model.pt",
              snn_ckpt)
        if not Path(SNN_CFG).exists():
            fetch("<anon>/spike-prophecy/scripts/standalone_snn_3l.yaml",
                  LOCAL / "standalone_snn_3l.yaml")
            cfg_path = str(LOCAL / "standalone_snn_3l.yaml")
        else:
            cfg_path = SNN_CFG
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        ckpt = torch.load(str(snn_ckpt), map_location="cpu",
                          weights_only=False)
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        model = StudentSNN.from_config(config, M_TRAIN)
        miss, unexp = model.load_state_dict(state, strict=False)
        _process_arch("SNN", model)
    except Exception as e:
        print(f"  SNN FAILED ({type(e).__name__}: {e})")

    # Save
    out_path = LOCAL / "ar_rollout.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    s3.upload_file(str(out_path), BUCKET, S3_OUT)
    print(f"Uploaded: s3://{BUCKET}/{S3_OUT}")


if __name__ == "__main__":
    main()
