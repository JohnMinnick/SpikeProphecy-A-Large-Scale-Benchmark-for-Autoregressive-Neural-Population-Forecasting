"""Class-balanced 16-class stim diagnostic (Jesus's prior question).

The Tab 4 16-class contrast number (Mamba 37.6%, H=10 sum 35.5%) sits
on top of a 26.1% (0,0) class prior — 26.1% of Steinmetz held-out
trials are no-stim. A trivially-always-(0,0) classifier already
hits 26.1%, so the apparent "6x chance" multiplier is misleading.

This script re-trains BOTH the Mamba-readout and H=10-sum
classifiers with sklearn class_weight='balanced' to remove the
prior and tells us whether the matched-context gain survives.

Outputs JSON to outputs/eval_local/diag_class_balanced_stim16.json
and prints a concise summary.
"""

import json
import logging
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.distill.multi_head_loss import contrast_to_class_index  # noqa: E402
from src.data.behavior_loader import extract_trial_stimuli  # noqa: E402

PRED_DIR = ROOT / "outputs" / "eval_local" / "behavioral_predictions" / "mamba"
CACHE_DIR = ROOT / "data" / "processed" / "combined_steinmetz_ibl_cache"
NWB_DIR = ROOT / "data" / "raw"
OUT_JSON = ROOT / "outputs" / "eval_local" / "diag_class_balanced_stim16.json"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("diag")
log.setLevel(logging.INFO)
warnings.filterwarnings("ignore")


def _split_eval_trials(trial_indices, sess_idx, frac=0.2, seed=42):
    unique = np.unique(trial_indices[trial_indices >= 0])
    rng = np.random.RandomState(seed + sess_idx)
    n_held = max(1, int(len(unique) * frac))
    eval_set = set(rng.choice(unique, size=n_held, replace=False).tolist())
    train_set = set(unique.tolist()) - eval_set
    return train_set, eval_set


def _build_h10_sum(counts, bin_indices, H=10):
    M, T = counts.shape
    feats = np.zeros((len(bin_indices), M), dtype=np.float32)
    for i, t in enumerate(bin_indices):
        t0 = max(0, t - H + 1)
        feats[i] = counts[:, t0:t + 1].sum(axis=1)
    return feats


def run_session(sess_idx, sess_info, nwb_files, cls_kwargs):
    """Returns (mamba_results_dict, baseline_results_dict) for stim
    16-class only, given a fitted-classifier-config dict."""
    n_bins_total = sess_info["num_bins"]
    bin_edges = np.arange(n_bins_total + 1) * 0.05
    nwb_path = nwb_files[sess_idx]
    beh = extract_trial_stimuli(str(nwb_path), bin_edges)
    trial_idx = beh["trial_index"]
    trial_active = beh["trial_active"] > 0.5
    train_set, eval_set = _split_eval_trials(trial_idx, sess_idx)
    train_bins = np.where(trial_active & np.array([
        int(trial_idx[b]) in train_set for b in range(len(trial_idx))
    ]))[0]
    eval_bins = np.where(trial_active & np.array([
        int(trial_idx[b]) in eval_set for b in range(len(trial_idx))
    ]))[0]
    if len(train_bins) < 50 or len(eval_bins) < 50:
        return None, None

    # ---- features
    feats = {}
    npz_path = PRED_DIR / f"session_{sess_idx:03d}.npz"
    if npz_path.exists():
        arr = np.load(npz_path)
        preds = arr["pred_rates"]
        split_start = int(arr["split_start_bin"])
        T_pred = preds.shape[1]
        train_pb = train_bins - split_start
        eval_pb = eval_bins - split_start
        tk = (train_pb >= 0) & (train_pb < T_pred)
        ek = (eval_pb >= 0) & (eval_pb < T_pred)
        if tk.sum() < 50 or ek.sum() < 50:
            feats["mamba"] = None
        else:
            feats["mamba"] = (
                preds[:, train_pb[tk]].T.astype(np.float32),
                preds[:, eval_pb[ek]].T.astype(np.float32),
                train_bins[tk], eval_bins[ek],
            )
    counts = np.load(CACHE_DIR / f"session_{sess_idx:03d}.npy")
    feats["h10sum"] = (
        _build_h10_sum(counts, train_bins),
        _build_h10_sum(counts, eval_bins),
        train_bins, eval_bins,
    )
    del counts

    out = {}
    for name, x in feats.items():
        if x is None:
            out[name] = None
            continue
        Xt, Xe, t_bins, e_bins = x
        sc = StandardScaler()
        Xtz = sc.fit_transform(Xt); Xez = sc.transform(Xe)
        left_t = beh["left_contrast"][t_bins].astype(np.float32)
        right_t = beh["right_contrast"][t_bins].astype(np.float32)
        left_e = beh["left_contrast"][e_bins].astype(np.float32)
        right_e = beh["right_contrast"][e_bins].astype(np.float32)
        y_t = contrast_to_class_index(
            torch.tensor(left_t), torch.tensor(right_t)
        ).numpy()
        y_e = contrast_to_class_index(
            torch.tensor(left_e), torch.tensor(right_e)
        ).numpy()
        try:
            clf = LogisticRegression(**cls_kwargs)
            clf.fit(Xtz, y_t)
            y_pred = clf.predict(Xez)
        except Exception:
            out[name] = None
            continue
        # Bin-level
        bin_acc = float((y_pred == y_e).mean())
        bin_bal = float(
            balanced_accuracy_score(y_e, y_pred)
        )
        # Trial-level vote
        eval_trial_ids = trial_idx[e_bins]
        n_correct = 0; n_t = 0
        per_trial_pred = []
        per_trial_true = []
        for tid in np.unique(eval_trial_ids):
            m = eval_trial_ids == tid
            if not m.any(): continue
            vote = int(np.bincount(y_pred[m]).argmax())
            truth = int(y_e[m][0])
            per_trial_pred.append(vote)
            per_trial_true.append(truth)
            n_correct += int(vote == truth); n_t += 1
        trial_acc = n_correct / max(n_t, 1)
        per_trial_pred = np.array(per_trial_pred)
        per_trial_true = np.array(per_trial_true)
        # Trial-level balanced acc
        trial_bal = float(
            balanced_accuracy_score(per_trial_true, per_trial_pred)
        )
        # Non-(0,0) trial accuracy
        non_zero = per_trial_true != 0
        trial_acc_nonzero = float(
            (per_trial_pred[non_zero] == per_trial_true[non_zero]).mean()
        ) if non_zero.any() else 0.0
        out[name] = {
            "bin_acc": bin_acc,
            "bin_balanced_acc": bin_bal,
            "trial_acc": trial_acc,
            "trial_balanced_acc": trial_bal,
            "trial_acc_non_zero_only": trial_acc_nonzero,
            "n_trials": n_t,
            "n_trials_non_zero": int(non_zero.sum()),
        }
    return out


def main():
    metadata = json.load(open(CACHE_DIR / "metadata.json"))
    n_steinmetz = sum(
        1 for s in metadata["sessions"] if s.get("source") == "steinmetz"
    )
    nwb_files = sorted(NWB_DIR.glob("Steinmetz2019_*.nwb"))

    configs = {
        "default": dict(C=1.0, max_iter=200, solver="lbfgs"),
        "balanced": dict(C=1.0, max_iter=200, solver="lbfgs",
                         class_weight="balanced"),
    }

    results = {cfg: {"mamba": [], "h10sum": []} for cfg in configs}
    t0 = time.time()
    for sess_idx in range(n_steinmetz):
        for cfg_name, kw in configs.items():
            res = run_session(sess_idx, metadata["sessions"][sess_idx],
                              nwb_files, kw)
            if res is None:
                continue
            for name in ("mamba", "h10sum"):
                if res.get(name) is not None:
                    results[cfg_name][name].append(
                        {"sess_idx": sess_idx, **res[name]}
                    )
        log.info("  s%03d done (%.0fs cum)", sess_idx, time.time() - t0)

    # ---- aggregate
    summary = {}
    for cfg in configs:
        summary[cfg] = {}
        for name in ("mamba", "h10sum"):
            recs = results[cfg][name]
            if not recs:
                summary[cfg][name] = None; continue
            n_total = sum(r["n_trials"] for r in recs)
            n_nz_total = sum(r["n_trials_non_zero"] for r in recs)
            agg = {
                "n_sessions": len(recs),
                "n_trials": n_total,
                "n_trials_non_zero": n_nz_total,
            }
            for k in ("bin_acc", "bin_balanced_acc", "trial_acc",
                      "trial_balanced_acc"):
                vals = np.array([r[k] for r in recs])
                weights = np.array([r["n_trials"] for r in recs], dtype=float)
                agg[k] = float(np.average(vals, weights=weights))
            # non-zero trial acc weighted by non-zero trial count
            vals = np.array([r["trial_acc_non_zero_only"] for r in recs])
            weights = np.array([r["n_trials_non_zero"] for r in recs],
                               dtype=float)
            mask = weights > 0
            agg["trial_acc_non_zero_only"] = float(
                np.average(vals[mask], weights=weights[mask])
            )
            summary[cfg][name] = agg

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "summary": summary,
            "per_session": results,
        }, f, indent=2)
    print(f"\nWrote {OUT_JSON}\n")
    print("=" * 78)
    print(f"{'config':<10s} {'feature':<7s} {'bin acc':>8s} {'bin bal':>8s} "
          f"{'trial':>8s} {'tr bal':>8s} {'non-(0,0)':>10s}")
    print("-" * 78)
    for cfg in configs:
        for name in ("mamba", "h10sum"):
            d = summary[cfg][name]
            if d is None: continue
            print(f"{cfg:<10s} {name:<7s} "
                  f"{100*d['bin_acc']:>7.2f}% {100*d['bin_balanced_acc']:>7.2f}% "
                  f"{100*d['trial_acc']:>7.2f}% {100*d['trial_balanced_acc']:>7.2f}% "
                  f"{100*d['trial_acc_non_zero_only']:>9.2f}%")
    print("=" * 78)
    # Gap analysis
    for cfg in configs:
        m = summary[cfg]["mamba"]; b = summary[cfg]["h10sum"]
        if m and b:
            print(f"\n{cfg}: Mamba - baseline (trial vote): "
                  f"{100*(m['trial_acc']-b['trial_acc']):+.2f} pp top-1, "
                  f"{100*(m['trial_balanced_acc']-b['trial_balanced_acc']):+.2f} pp balanced, "
                  f"{100*(m['trial_acc_non_zero_only']-b['trial_acc_non_zero_only']):+.2f} pp non-(0,0)")


if __name__ == "__main__":
    main()
