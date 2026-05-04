"""Three figure mockups requested by Jesus (NOT for paper unless approved).

  J1  Pipeline schematic
        Block diagram: Spikes -> H=10 history window -> Mamba ->
        Predicted rates -> Per-session linear readout -> Behavior.
        Annotated with dimensions, latency, calibration block size.

  J2  Per-session scatter (Mamba vs H=10 sum baseline)
        Three panels (response / stim 16-class / side). Each session
        a point. y=x diagonal. Color = win/loss vs baseline.
        Decomposes Tab 4 across sessions.

  J3  Dual confusion matrix (Mamba vs H=10 sum baseline)
        Side-by-side 16-class confusion matrices for Mamba and the
        matched H=10 sum baseline.  Addresses Limitation 4 ("5/16
        conditions recoverable") by visualising which contrast
        conditions each model gets right.

Outputs in docs/neurips_neurocog/figures/mockups/jesus_*.{png,pdf}.
~30-60 min compute total (J3 trains 39 H=10 sum classifiers).
"""

import json
import logging
import sys
import time
import warnings
from pathlib import Path

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.distill.multi_head_loss import (  # noqa: E402
    contrast_to_class_index,
    CONTRAST_LEVELS,
)
from src.data.behavior_loader import extract_trial_stimuli  # noqa: E402

PRED_DIR = ROOT / "outputs" / "eval_local" / "behavioral_predictions" / "mamba"
CACHE_DIR = ROOT / "data" / "processed" / "combined_steinmetz_ibl_cache"
CACHE_META = CACHE_DIR / "metadata.json"
NWB_DIR = ROOT / "data" / "raw"
EVAL_DIR = ROOT / "outputs" / "eval_local"
OUT = ROOT / "docs" / "neurips_neurocog" / "figures" / "mockups"
OUT.mkdir(parents=True, exist_ok=True)

# Coherent palette
COLOR_MAMBA = "#2c7bb6"     # blue
COLOR_BASE = "#bbbbbb"      # neutral grey
COLOR_WIN = "#2c7bb6"
COLOR_LOSS = "#d7191c"
COLOR_RESP = "#2c7bb6"
COLOR_STIM = "#fdae61"
COLOR_SIDE = "#1a9641"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mockup_jesus")
log.setLevel(logging.INFO)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Shared training (Mamba + H=10 sum) — saves per-trial predictions
# ---------------------------------------------------------------------------
def _split_eval_trials(trial_indices, sess_idx, frac=0.2, seed=42):
    unique = np.unique(trial_indices[trial_indices >= 0])
    rng = np.random.RandomState(seed + sess_idx)
    n_held = max(1, int(len(unique) * frac))
    eval_set = set(rng.choice(unique, size=n_held, replace=False).tolist())
    train_set = set(unique.tolist()) - eval_set
    return train_set, eval_set


def _build_h10_sum_features(counts, bin_indices, H=10):
    M, T = counts.shape
    feats = np.zeros((len(bin_indices), M), dtype=np.float32)
    for i, t in enumerate(bin_indices):
        t0 = max(0, t - H + 1)
        feats[i] = counts[:, t0:t + 1].sum(axis=1)
    return feats


def train_per_session(target_features="mamba"):
    """Per-session train+eval, saving per-trial predictions for the
    16-class stim target (and resp/side bin-level for the scatter).

    target_features in {'mamba', 'h10_sum'}.
    """
    metadata = json.load(open(CACHE_META))
    n_steinmetz = sum(
        1 for s in metadata["sessions"] if s.get("source") == "steinmetz"
    )
    nwb_files = sorted(NWB_DIR.glob("Steinmetz2019_*.nwb"))

    # Per-trial: collect (predicted_stim, true_stim) over held-out trials
    trial_pred_stim, trial_true_stim = [], []
    # Per-session bin acc for resp/stim/side (used by scatter)
    per_session = []  # list of {session_idx, resp_acc, stim_acc, side_acc}

    t0 = time.time()
    for sess_idx in range(n_steinmetz):
        sess_info = metadata["sessions"][sess_idx]
        n_bins_total = sess_info["num_bins"]
        bin_edges = np.arange(n_bins_total + 1) * 0.05
        nwb_path = nwb_files[sess_idx]
        beh = extract_trial_stimuli(str(nwb_path), bin_edges)
        trial_idx = beh["trial_index"]
        trial_active = beh["trial_active"] > 0.5
        train_set, eval_set = _split_eval_trials(trial_idx, sess_idx)
        train_bins = np.where(
            trial_active & np.array([
                int(trial_idx[b]) in train_set
                for b in range(len(trial_idx))
            ])
        )[0]
        eval_bins = np.where(
            trial_active & np.array([
                int(trial_idx[b]) in eval_set
                for b in range(len(trial_idx))
            ])
        )[0]
        if len(train_bins) < 50 or len(eval_bins) < 50:
            continue

        # Build features depending on target
        if target_features == "mamba":
            npz_path = PRED_DIR / f"session_{sess_idx:03d}.npz"
            if not npz_path.exists():
                continue
            arr = np.load(npz_path)
            preds = arr["pred_rates"]               # (M_i, T_full)
            split_start = int(arr["split_start_bin"])
            T_pred = preds.shape[1]
            train_pred_bins = train_bins - split_start
            eval_pred_bins = eval_bins - split_start
            train_keep = (train_pred_bins >= 0) & (train_pred_bins < T_pred)
            eval_keep = (eval_pred_bins >= 0) & (eval_pred_bins < T_pred)
            train_pred_bins = train_pred_bins[train_keep]
            eval_pred_bins = eval_pred_bins[eval_keep]
            train_bins_use = train_bins[train_keep]
            eval_bins_use = eval_bins[eval_keep]
            X_train = preds[:, train_pred_bins].T.astype(np.float32)
            X_eval = preds[:, eval_pred_bins].T.astype(np.float32)
        else:  # h10_sum
            counts = np.load(CACHE_DIR / f"session_{sess_idx:03d}.npy")
            X_train = _build_h10_sum_features(counts, train_bins)
            X_eval = _build_h10_sum_features(counts, eval_bins)
            train_bins_use = train_bins
            eval_bins_use = eval_bins
            del counts

        scaler = StandardScaler()
        X_train_z = scaler.fit_transform(X_train)
        X_eval_z = scaler.transform(X_eval)

        # Targets
        left_t = beh["left_contrast"][train_bins_use]
        right_t = beh["right_contrast"][train_bins_use]
        left_e = beh["left_contrast"][eval_bins_use]
        right_e = beh["right_contrast"][eval_bins_use]
        y_resp_t = (beh["response_choice"][train_bins_use] + 1).astype(np.int64)
        y_resp_e = (beh["response_choice"][eval_bins_use] + 1).astype(np.int64)
        y_stim_t = contrast_to_class_index(
            torch.tensor(left_t.astype(np.float32)),
            torch.tensor(right_t.astype(np.float32)),
        ).numpy()
        y_stim_e = contrast_to_class_index(
            torch.tensor(left_e.astype(np.float32)),
            torch.tensor(right_e.astype(np.float32)),
        ).numpy()
        y_side_t = (np.sign(right_t - left_t).astype(np.int64) + 1)
        y_side_e = (np.sign(right_e - left_e).astype(np.int64) + 1)

        clf_resp = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
        clf_resp.fit(X_train_z, y_resp_t)
        clf_stim = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
        clf_stim.fit(X_train_z, y_stim_t)
        clf_side = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
        clf_side.fit(X_train_z, y_side_t)

        yr_p = clf_resp.predict(X_eval_z)
        ys_p = clf_stim.predict(X_eval_z)
        yd_p = clf_side.predict(X_eval_z)

        per_session.append({
            "session_idx": sess_idx,
            "n_eval_bins": int(len(eval_bins_use)),
            "resp_acc": float((yr_p == y_resp_e).mean()),
            "stim_acc": float((ys_p == y_stim_e).mean()),
            "side_acc": float((yd_p == y_side_e).mean()),
        })

        # Trial-level majority vote on the 16-class stim target
        eval_trial_ids = trial_idx[eval_bins_use]
        for tid in np.unique(eval_trial_ids):
            m = eval_trial_ids == tid
            if not m.any():
                continue
            trial_pred_stim.append(int(np.bincount(ys_p[m]).argmax()))
            trial_true_stim.append(int(y_stim_e[m][0]))

        log.info(
            "  %s s%03d done (%.0fs cum)",
            target_features, sess_idx, time.time() - t0,
        )

    return {
        "per_session": per_session,
        "trial_pred_stim": np.array(trial_pred_stim),
        "trial_true_stim": np.array(trial_true_stim),
    }


# ---------------------------------------------------------------------------
# J1 — pipeline schematic
# ---------------------------------------------------------------------------
def fig_J1_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 4.0), constrained_layout=True)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, label, sub, fc="white", ec="0.3", text_c="0.1",
            zorder=2):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
            fc=fc, ec=ec, lw=1.4, zorder=zorder,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=text_c)
        if sub:
            ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                    fontsize=8, color="0.35", style="italic")

    def arrow(x1, y1, x2, y2, color="0.4", label=None, label_off=4):
        ax.annotate(
            "", xy=(x2, y1), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.4),
        )
        if label:
            ax.text((x1 + x2) / 2, y1 + label_off, label, ha="center",
                    va="bottom", fontsize=8, color="0.3", style="italic")

    # Boxes (left to right)
    y0, h = 38, 28
    box(2,   y0, 14, h,
        "Spike counts",
        "$M$ neurons\n50 ms bins",
        fc="#f4f6f9")
    box(20,  y0, 14, h,
        "History window",
        "$H{=}10$ bins\n(500 ms)",
        fc="#e7eef5")
    box(38,  y0, 18, h,
        "Mamba forecaster",
        "selective SSM\n1.95M params\n$\\leq$6.4 ms / batch",
        fc=COLOR_MAMBA, text_c="white")
    box(60,  y0, 16, h,
        "Predicted rates",
        "$M$ neurons\nnext bin",
        fc="#e7eef5")
    box(80,  y0, 18, h,
        "Linear readout",
        "per session\n100--150 trials\nto calibrate",
        fc="#fcefdc")

    # Arrows
    arrow(16, y0 + h / 2, 20, y0 + h / 2)
    arrow(34, y0 + h / 2, 38, y0 + h / 2)
    arrow(56, y0 + h / 2, 60, y0 + h / 2)
    arrow(76, y0 + h / 2, 80, y0 + h / 2)

    # Three behavioral output bubbles (right of linear readout)
    out_x = 80
    out_y = y0 - 14
    for dy, lbl, c in [
        (-3.5, "Response (3-class)", COLOR_RESP),
        (-9.0, "Stimulus contrast (16-class)", COLOR_STIM),
        (-14.5, "Stimulus side (3-class)", COLOR_SIDE),
    ]:
        ax.annotate(
            "", xy=(out_x + 9, out_y - dy + 4), xytext=(out_x + 9, y0 - 1),
            arrowprops=dict(arrowstyle="-", color="0.6", lw=0.7,
                            connectionstyle="arc3,rad=0.0"),
        )
        ax.text(
            out_x + 22, out_y - dy + 4, lbl, ha="left", va="center",
            fontsize=9, color=c, fontweight="bold",
        )

    # Top label: full pipeline latency
    ax.text(
        50, 88,
        "One forward pass produces a population-rate forecast and a "
        "per-session linear behavioral readout",
        ha="center", va="center", fontsize=11, fontweight="bold",
    )
    ax.text(
        50, 81,
        "Trained only on next-step Poisson NLL loss; behavioral labels "
        "never seen at training time",
        ha="center", va="center", fontsize=9, color="0.4", style="italic",
    )

    # Bottom annotations
    ax.text(
        50, 18,
        "Total per-bin latency $\\approx$ Mamba forward pass + "
        "closed-form logistic regression $\\ll$ 50 ms bin budget",
        ha="center", va="center", fontsize=9, color="0.3",
    )

    fig.suptitle(
        "J1 — Pipeline schematic (single Mamba forecaster, two outputs)",
        fontsize=11, y=1.02,
    )
    fig.savefig(OUT / "jesus_J1_pipeline.png", dpi=160, bbox_inches="tight")
    fig.savefig(OUT / "jesus_J1_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote jesus_J1_pipeline.{png,pdf}")


# ---------------------------------------------------------------------------
# J2 — per-session scatter, Mamba vs H=10 sum, three panels
# ---------------------------------------------------------------------------
def fig_J2_per_session_scatter(mamba_rec, base_rec):
    mamba_per = {r["session_idx"]: r for r in mamba_rec["per_session"]}
    base_per = {r["session_idx"]: r for r in base_rec["per_session"]}
    sess_ids = sorted(set(mamba_per) & set(base_per))

    targets = [
        ("resp_acc", "Response (3-class)", 1 / 3),
        ("stim_acc", "Stimulus contrast (16-class)", 1 / 16),
        ("side_acc", "Stimulus side (3-class)", 1 / 3),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8),
                             constrained_layout=True)
    for ax, (key, title, chance) in zip(axes, targets):
        x = np.array([base_per[s][key] for s in sess_ids])
        y = np.array([mamba_per[s][key] for s in sess_ids])
        win = y > x
        lo = min(x.min(), y.min(), chance) - 0.03
        hi = max(x.max(), y.max()) + 0.03
        ax.plot([lo, hi], [lo, hi], color="0.5", linestyle="--", linewidth=1)
        ax.axvline(chance, color="0.85", linewidth=0.6, zorder=0)
        ax.axhline(chance, color="0.85", linewidth=0.6, zorder=0)
        ax.scatter(
            x[win], y[win], s=30, c=COLOR_WIN, edgecolor="white",
            linewidth=0.5, zorder=3,
            label=f"Mamba wins ({int(win.sum())}/{len(win)})",
        )
        ax.scatter(
            x[~win], y[~win], s=30, c=COLOR_LOSS, edgecolor="white",
            linewidth=0.5, zorder=3,
            label=f"Baseline wins ({int((~win).sum())}/{len(win)})",
        )
        delta = (y - x).mean() * 100
        ax.text(
            0.04, 0.96, f"Mean $\\Delta$ = {delta:+.1f} pp",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6),
        )
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel("H=10 sum baseline (bin acc)")
        ax.set_ylabel("Mamba seed 42 (bin acc)")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    fig.suptitle(
        "J2 — Per-session matched-context comparison "
        "(Steinmetz 39, Tab. 4 decomposed across sessions)",
        fontsize=11, y=1.04,
    )
    fig.savefig(OUT / "jesus_J2_per_session_scatter.png", dpi=160,
                bbox_inches="tight")
    fig.savefig(OUT / "jesus_J2_per_session_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote jesus_J2_per_session_scatter.{png,pdf}")


# ---------------------------------------------------------------------------
# J3 — dual confusion matrix, Mamba vs H=10 sum
# ---------------------------------------------------------------------------
def _confusion_matrix(pred, true, n_classes=16):
    cm = np.zeros((n_classes, n_classes), dtype=np.float32)
    for p, t in zip(pred, true):
        cm[t, p] += 1
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    return cm


def fig_J3_dual_confusion(mamba_rec, base_rec):
    cm_mamba = _confusion_matrix(
        mamba_rec["trial_pred_stim"], mamba_rec["trial_true_stim"],
    )
    cm_base = _confusion_matrix(
        base_rec["trial_pred_stim"], base_rec["trial_true_stim"],
    )

    L = list(CONTRAST_LEVELS)
    labels = [f"({L[c // 4]:.2g}, {L[c % 4]:.2g})" for c in range(16)]

    vmax = max(cm_mamba.max(), cm_base.max())
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), constrained_layout=True)

    for ax, cm, name, n_trials in [
        (axes[0], cm_mamba, "Mamba seed 42",
         len(mamba_rec["trial_true_stim"])),
        (axes[1], cm_base, "H=10 sum baseline",
         len(base_rec["trial_true_stim"])),
    ]:
        im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=vmax,
                       aspect="equal")
        ax.set_xticks(range(16)); ax.set_yticks(range(16))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Predicted contrast (left, right)")
        ax.set_ylabel("True contrast (left, right)")
        diag = np.diag(cm).mean()
        top1 = float(np.diag(cm).sum() / max(cm.sum(), 1))  # not quite
        # better top-1: from raw counts
        ax.set_title(
            f"{name}\n"
            f"diagonal mean = {diag:.2f} | trial-vote top-1 acc = ",
            fontsize=10,
        )

    # Compute top-1 trial-vote accuracy from raw counts for the title
    top1_mamba = float(
        (mamba_rec["trial_pred_stim"] == mamba_rec["trial_true_stim"]).mean()
    )
    top1_base = float(
        (base_rec["trial_pred_stim"] == base_rec["trial_true_stim"]).mean()
    )
    axes[0].set_title(
        f"Mamba seed 42\n"
        f"diagonal mean = {np.diag(cm_mamba).mean():.2f} | "
        f"top-1 trial-vote acc = {top1_mamba:.2f}",
        fontsize=10,
    )
    axes[1].set_title(
        f"H=10 sum baseline\n"
        f"diagonal mean = {np.diag(cm_base).mean():.2f} | "
        f"top-1 trial-vote acc = {top1_base:.2f}",
        fontsize=10,
    )

    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                 label="P(predicted | true)")
    fig.suptitle(
        f"J3 — 16-class contrast confusion (Steinmetz 39, "
        f"{len(mamba_rec['trial_true_stim']):,} held-out trials)",
        fontsize=11, y=1.02,
    )
    fig.savefig(OUT / "jesus_J3_dual_confusion.png", dpi=160,
                bbox_inches="tight")
    fig.savefig(OUT / "jesus_J3_dual_confusion.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote jesus_J3_dual_confusion.{png,pdf}")


# ---------------------------------------------------------------------------
def main():
    print("\n--- J1: Pipeline schematic ---")
    fig_J1_pipeline()

    print("\n--- J2/J3 data: training Mamba per-session decoders ---")
    mamba = train_per_session(target_features="mamba")
    print(f"  Mamba trials: {len(mamba['trial_true_stim']):,}")

    print("\n--- J2/J3 data: training H=10 sum per-session decoders ---")
    base = train_per_session(target_features="h10_sum")
    print(f"  H=10 sum trials: {len(base['trial_true_stim']):,}")

    print("\n--- J2: per-session scatter ---")
    fig_J2_per_session_scatter(mamba, base)

    print("\n--- J3: dual confusion matrix ---")
    fig_J3_dual_confusion(mamba, base)

    print(f"\nDone. Mockups in {OUT}/")


if __name__ == "__main__":
    main()
