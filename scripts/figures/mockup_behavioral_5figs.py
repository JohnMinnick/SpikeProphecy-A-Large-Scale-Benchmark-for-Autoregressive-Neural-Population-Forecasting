"""Five behavioral-decoding figure mockups (NOT for paper).

All five emphasize the BEHAVIORAL READOUT rather than architecture
ranking, keeping the paper's application/scientific framing intact:

  A  Decoding time-course around stimulus onset
       PETH-style: trial-aligned bin-by-bin decoder accuracy
       across the [-0.5 s, +1.5 s] window. Three lines (resp / stim 16 /
       side) with shaded 95% CIs, vertical dashed line at onset.
       Speaks to closed-loop deployment (when does decoding peak?).

  B  16-class contrast confusion matrix
       Predicted vs true left-right contrast pair, sorted by
       (left_contrast, right_contrast). Visualizes confusion structure:
       are mistakes adjacent contrasts (sensible) or random?

  C  Psychometric / difficulty curve
       Decoder accuracy as a function of signed contrast difference
       (right minus left). Shows decoder accuracy scales with task
       difficulty.

  D  Trial-aligned population-rate heatmap
       Each row = one trial, time-aligned to onset, color = predicted
       population rate. Trials sorted by behavior class (left choice /
       no-go / right choice). Shows trial-locked population dynamics.

  E  Brain-region-resolved decoding
       Per-region decoder accuracy when the input is restricted to
       neurons in a single Allen CCF region. Aggregated across the 39
       Steinmetz sessions, weighted by per-session neuron count.

Outputs to docs/neurips_neurocog/figures/mockups/behav_*.{png,pdf}.
Run with the project venv: ~30-60 min compute end-to-end.
"""

import json
import logging
import sys
import time
import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.distill.multi_head_loss import (  # noqa: E402
    CONTRAST_LEVELS,
    contrast_to_class_index,
)
from src.data.behavior_loader import extract_trial_stimuli  # noqa: E402

PRED_DIR = ROOT / "outputs" / "eval_local" / "behavioral_predictions" / "mamba"
NWB_DIR = ROOT / "data" / "raw"
CACHE_META = (
    ROOT / "data" / "processed" / "combined_steinmetz_ibl_cache" / "metadata.json"
)
REGION_MAPPING = (
    ROOT / "outputs" / "eval_analysis" / "brain_region_mapping.json"
)
OUT = ROOT / "docs" / "neurips_neurocog" / "figures" / "mockups"
OUT.mkdir(parents=True, exist_ok=True)

# NEDS-inspired application palette
COLOR_RESP = "#2c7bb6"      # blue, response
COLOR_STIM = "#fdae61"      # warm orange, stim 16-class
COLOR_SIDE = "#1a9641"      # green, side
COLOR_REF = "0.55"          # neutral grey

BIN_S = 0.05
WIN_PRE_S = 0.5             # 500 ms pre-stim
WIN_POST_S = 1.5            # 1500 ms post-stim
WIN_PRE_BINS = int(round(WIN_PRE_S / BIN_S))
WIN_POST_BINS = int(round(WIN_POST_S / BIN_S))
WIN_LEN = WIN_PRE_BINS + WIN_POST_BINS  # 40 bins total

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mockup_behav")
log.setLevel(logging.INFO)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def stim_on_times(nwb_path: str) -> np.ndarray:
    """Per-trial stimulus onset times (seconds)."""
    with h5py.File(nwb_path, "r") as f:
        return f["intervals/trials/visual_stimulus_time"][:]


def build_trial_records():
    """Train per-session decoders, return aggregated trial records.

    Returns a dict keyed by behavioral target ('resp' / 'stim' / 'side')
    plus auxiliary arrays the mockups need.
    """
    metadata = json.load(open(CACHE_META))
    n_steinmetz = sum(
        1 for s in metadata["sessions"] if s.get("source") == "steinmetz"
    )
    nwb_files = sorted(NWB_DIR.glob("Steinmetz2019_*.nwb"))

    trial_pred_resp, trial_true_resp = [], []
    trial_pred_stim, trial_true_stim = [], []
    trial_pred_side, trial_true_side = [], []
    trial_signed_contrast = []
    trial_session = []

    # Trial-aligned per-bin predictions (resp / stim / side) collected per
    # held-out trial.  Shape after vstack: (n_trials_total, WIN_LEN).
    aligned_resp_pred, aligned_resp_true = [], []
    aligned_stim_pred, aligned_stim_true = [], []
    aligned_side_pred, aligned_side_true = [], []
    aligned_pop_rate = []     # population sum across neurons in window

    # For mockup E (region decoding) we need per-region accuracy. Defer.
    region_records = []

    side_map_t = torch.tensor(
        [
            1 if CONTRAST_LEVELS[c // 4] > CONTRAST_LEVELS[c % 4]
            else (2 if CONTRAST_LEVELS[c % 4] > CONTRAST_LEVELS[c // 4] else 0)
            for c in range(16)
        ],
        dtype=torch.long,
    )  # noqa: F841 (kept for parity / future use)

    region_map = json.load(open(REGION_MAPPING))["sessions"]

    t0 = time.time()
    for sess_idx in range(n_steinmetz):
        sess_info = metadata["sessions"][sess_idx]
        npz_path = PRED_DIR / f"session_{sess_idx:03d}.npz"
        if not npz_path.exists():
            continue
        arr = np.load(npz_path)
        preds = arr["pred_rates"]                  # (M_i, T_full)
        split_start = int(arr["split_start_bin"])  # legacy schema
        m_i = preds.shape[0]
        n_bins_total = sess_info["num_bins"]
        bin_edges = np.arange(n_bins_total + 1) * BIN_S

        nwb_path = nwb_files[sess_idx]
        beh = extract_trial_stimuli(str(nwb_path), bin_edges)
        stim_on = stim_on_times(str(nwb_path))     # per-trial seconds
        trial_active = beh["trial_active"] > 0.5
        trial_idx = beh["trial_index"]

        # Active bins fall in val/full split; predictions cover bins
        # [split_start, split_start + T_pred). Map session bin -> column
        # in `preds` by subtracting split_start.
        T_pred = preds.shape[1]
        active_session_bins = np.where(trial_active)[0]
        active_pred_bins = active_session_bins - split_start
        keep = (active_pred_bins >= 0) & (active_pred_bins < T_pred)
        active_session_bins = active_session_bins[keep]
        active_pred_bins = active_pred_bins[keep]
        if len(active_session_bins) < 200:
            continue

        # Trial-level 80/20 holdout (seed 42, matches eval pipeline)
        unique_trials = np.unique(trial_idx[trial_idx >= 0])
        rng = np.random.RandomState(42 + sess_idx)
        n_held = max(1, int(len(unique_trials) * 0.2))
        eval_trials = set(
            rng.choice(unique_trials, size=n_held, replace=False).tolist()
        )
        train_trials = set(unique_trials.tolist()) - eval_trials

        train_mask = np.array([
            int(trial_idx[b]) in train_trials for b in active_session_bins
        ])
        eval_mask = ~train_mask

        train_bins_pred = active_pred_bins[train_mask]
        train_bins_sess = active_session_bins[train_mask]
        eval_bins_pred = active_pred_bins[eval_mask]
        eval_bins_sess = active_session_bins[eval_mask]

        if len(train_bins_pred) < 50 or len(eval_bins_pred) < 50:
            continue

        # Build features (X) at each trial-active bin
        X_train = preds[:, train_bins_pred].T.astype(np.float32)
        X_eval = preds[:, eval_bins_pred].T.astype(np.float32)
        scaler = StandardScaler()
        X_train_z = scaler.fit_transform(X_train)
        X_eval_z = scaler.transform(X_eval)

        # Targets at each bin (from beh dict)
        left_t = beh["left_contrast"][train_bins_sess].astype(np.float32)
        right_t = beh["right_contrast"][train_bins_sess].astype(np.float32)
        left_e = beh["left_contrast"][eval_bins_sess].astype(np.float32)
        right_e = beh["right_contrast"][eval_bins_sess].astype(np.float32)
        y_resp_t = (beh["response_choice"][train_bins_sess] + 1).astype(np.int64)
        y_resp_e = (beh["response_choice"][eval_bins_sess] + 1).astype(np.int64)
        y_stim_t = contrast_to_class_index(
            torch.tensor(left_t), torch.tensor(right_t),
        ).numpy()
        y_stim_e = contrast_to_class_index(
            torch.tensor(left_e), torch.tensor(right_e),
        ).numpy()
        y_side_t = (np.sign(right_t - left_t).astype(np.int64) + 1)
        y_side_e = (np.sign(right_e - left_e).astype(np.int64) + 1)

        # --- train classifiers ---
        clf_resp = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
        clf_resp.fit(X_train_z, y_resp_t)
        clf_stim = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
        clf_stim.fit(X_train_z, y_stim_t)
        clf_side = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
        clf_side.fit(X_train_z, y_side_t)

        # --- per-trial predictions on the eval bins ---
        # Aggregate by trial (majority vote across bins) for the
        # confusion / psychometric / scatter mockups.
        eval_trial_ids = trial_idx[eval_bins_sess]
        unique_eval_trials = np.unique(eval_trial_ids)
        for tid in unique_eval_trials:
            m = eval_trial_ids == tid
            if not m.any():
                continue
            yr_p = clf_resp.predict(X_eval_z[m])
            ys_p = clf_stim.predict(X_eval_z[m])
            yd_p = clf_side.predict(X_eval_z[m])
            trial_pred_resp.append(int(np.bincount(yr_p).argmax()))
            trial_pred_stim.append(int(np.bincount(ys_p).argmax()))
            trial_pred_side.append(int(np.bincount(yd_p).argmax()))
            trial_true_resp.append(int(y_resp_e[m][0]))
            trial_true_stim.append(int(y_stim_e[m][0]))
            trial_true_side.append(int(y_side_e[m][0]))
            l = float(beh["left_contrast"][eval_bins_sess][m][0])
            r = float(beh["right_contrast"][eval_bins_sess][m][0])
            trial_signed_contrast.append(r - l)
            trial_session.append(sess_idx)

        # --- trial-aligned bin-by-bin window ---
        # For each held-out trial, take the [-WIN_PRE, +WIN_POST] window
        # around stimOn, predict bin-by-bin, and stack.
        for tid in unique_eval_trials:
            t_start = stim_on[tid]
            on_bin = int(round(t_start / BIN_S))
            lo = on_bin - WIN_PRE_BINS
            hi = on_bin + WIN_POST_BINS
            if lo - split_start < 0 or hi - split_start > T_pred:
                continue
            X_win = preds[:, lo - split_start:hi - split_start].T.astype(
                np.float32
            )
            X_win_z = scaler.transform(X_win)
            yr_p = clf_resp.predict(X_win_z)
            ys_p = clf_stim.predict(X_win_z)
            yd_p = clf_side.predict(X_win_z)
            yr_true = int(beh["response_choice"][on_bin] + 1)
            ys_true = int(contrast_to_class_index(
                torch.tensor([beh["left_contrast"][on_bin]]),
                torch.tensor([beh["right_contrast"][on_bin]]),
            ).item())
            yd_true = int(np.sign(
                beh["right_contrast"][on_bin] - beh["left_contrast"][on_bin]
            )) + 1
            aligned_resp_pred.append(yr_p)
            aligned_resp_true.append(yr_true)
            aligned_stim_pred.append(ys_p)
            aligned_stim_true.append(ys_true)
            aligned_side_pred.append(yd_p)
            aligned_side_true.append(yd_true)
            # Population rate in window (sum across all neurons)
            pop = preds[:, lo - split_start:hi - split_start].sum(axis=0)
            aligned_pop_rate.append(pop)

        # --- region-resolved decoding (mockup E) ---
        sess_key = str(sess_idx)
        if sess_key in region_map:
            neuron_regions = region_map[sess_key].get("neuron_regions", [])
            if len(neuron_regions) == m_i:
                neuron_regions = np.array(neuron_regions)
                regions_present = sorted(set(neuron_regions))
                for region in regions_present:
                    sel = neuron_regions == region
                    n_sel = int(sel.sum())
                    if n_sel < 4:
                        continue
                    Xr_t = preds[sel, :][:, train_bins_pred].T
                    Xr_e = preds[sel, :][:, eval_bins_pred].T
                    sr = StandardScaler()
                    Xr_t = sr.fit_transform(Xr_t)
                    Xr_e = sr.transform(Xr_e)
                    accs = {}
                    for tgt, yt, ye in (
                        ("resp", y_resp_t, y_resp_e),
                        ("stim", y_stim_t, y_stim_e),
                        ("side", y_side_t, y_side_e),
                    ):
                        try:
                            cl = LogisticRegression(
                                C=1.0, max_iter=200, solver="lbfgs",
                            )
                            cl.fit(Xr_t, yt)
                            accs[tgt] = float(
                                (cl.predict(Xr_e) == ye).mean()
                            )
                        except Exception:
                            accs[tgt] = 0.0
                    region_records.append({
                        "session_idx": sess_idx,
                        "region": region,
                        "n_neurons": n_sel,
                        **accs,
                    })

        log.info(
            "  s%03d done | %d eval trials, %.0fs cum",
            sess_idx, len(unique_eval_trials), time.time() - t0,
        )

    return {
        "trial_pred_resp": np.array(trial_pred_resp),
        "trial_true_resp": np.array(trial_true_resp),
        "trial_pred_stim": np.array(trial_pred_stim),
        "trial_true_stim": np.array(trial_true_stim),
        "trial_pred_side": np.array(trial_pred_side),
        "trial_true_side": np.array(trial_true_side),
        "trial_signed_contrast": np.array(trial_signed_contrast),
        "trial_session": np.array(trial_session),
        "aligned_resp_pred": np.array(aligned_resp_pred),
        "aligned_resp_true": np.array(aligned_resp_true),
        "aligned_stim_pred": np.array(aligned_stim_pred),
        "aligned_stim_true": np.array(aligned_stim_true),
        "aligned_side_pred": np.array(aligned_side_pred),
        "aligned_side_true": np.array(aligned_side_true),
        "aligned_pop_rate": np.array(aligned_pop_rate),
        "region_records": region_records,
    }


# ---------------------------------------------------------------------------
# Mockup A — decoding time-course around stim onset
# ---------------------------------------------------------------------------
def mockup_A(rec):
    pred_r = rec["aligned_resp_pred"]   # (n_trials, WIN_LEN)
    true_r = rec["aligned_resp_true"]
    pred_s = rec["aligned_stim_pred"]
    true_s = rec["aligned_stim_true"]
    pred_d = rec["aligned_side_pred"]
    true_d = rec["aligned_side_true"]

    def acc_curve(pred, true):
        n_trials, n_bins = pred.shape
        eq = (pred == true[:, None]).astype(np.float32)
        mean = eq.mean(axis=0)
        # Bootstrap CI across trials
        rng = np.random.default_rng(0)
        n_boot = 500
        boot = np.empty((n_boot, n_bins))
        for i in range(n_boot):
            idx = rng.integers(0, n_trials, size=n_trials)
            boot[i] = eq[idx].mean(axis=0)
        lo = np.percentile(boot, 2.5, axis=0)
        hi = np.percentile(boot, 97.5, axis=0)
        return mean, lo, hi

    t = (np.arange(WIN_LEN) - WIN_PRE_BINS) * BIN_S  # seconds, 0 = onset
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    for pred, true, c, lab, chance in [
        (pred_r, true_r, COLOR_RESP, "Response (3-class)", 1 / 3),
        (pred_s, true_s, COLOR_STIM, "Stim contrast (16-class)", 1 / 16),
        (pred_d, true_d, COLOR_SIDE, "Stimulus side (3-class)", 1 / 3),
    ]:
        m, lo, hi = acc_curve(pred, true)
        ax.fill_between(t, lo, hi, color=c, alpha=0.18, linewidth=0)
        ax.plot(t, m, color=c, linewidth=2.0, label=lab)
        ax.axhline(chance, color=c, linestyle=":", linewidth=0.6, alpha=0.5)
    ax.axvline(0, color="0.3", linestyle="--", linewidth=1.0,
               label="stimulus onset")
    ax.set_xlabel("Time relative to stimulus onset (s)")
    ax.set_ylabel("Decoder accuracy (per-bin)")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, max(0.9, max(true_r.max() if len(true_r) else 1.0, 1.0)))
    ax.set_title(
        "A — Decoding time-course around stimulus onset\n"
        f"Steinmetz 39 sessions, {pred_r.shape[0]:,} held-out trials, "
        "Mamba seed 42 predicted rates",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.savefig(OUT / "behav_A_time_course.png", dpi=160, bbox_inches="tight")
    fig.savefig(OUT / "behav_A_time_course.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote behav_A_time_course.{png,pdf}")


# ---------------------------------------------------------------------------
# Mockup B — 16-class confusion matrix
# ---------------------------------------------------------------------------
def mockup_B(rec):
    pred = rec["trial_pred_stim"]
    true = rec["trial_true_stim"]
    cm = np.zeros((16, 16), dtype=np.float32)
    for p, t in zip(pred, true):
        cm[t, p] += 1
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    # Class index -> (left, right) contrast pair label
    levels = [0.0, 0.0625, 0.125, 0.25, 1.0]
    # Wait — CONTRAST_LEVELS in src is 4-element [0, 0.25, 0.5, 1.0]?
    # use the actual import
    L = list(CONTRAST_LEVELS)
    labels = [f"({L[c // 4]:.2g}, {L[c % 4]:.2g})" for c in range(16)]

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=cm.max())
    ax.set_xticks(range(16)); ax.set_yticks(range(16))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted contrast (left, right)")
    ax.set_ylabel("True contrast (left, right)")

    # Diagonal correct overlay
    diag_acc = np.diag(cm).mean()
    ax.set_title(
        "B — 16-class contrast confusion matrix\n"
        f"Steinmetz 39, {len(true):,} held-out trials | "
        f"diagonal mean = {diag_acc:.2f}, top-1 acc = {(pred == true).mean():.2f}",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, label="P(predicted | true)")
    fig.savefig(OUT / "behav_B_confusion.png", dpi=160, bbox_inches="tight")
    fig.savefig(OUT / "behav_B_confusion.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote behav_B_confusion.{png,pdf}")


# ---------------------------------------------------------------------------
# Mockup C — psychometric / difficulty curve
# ---------------------------------------------------------------------------
def mockup_C(rec):
    diff = rec["trial_signed_contrast"]
    # bin signed-contrast into ~7 quantile buckets centered at 0
    bins = np.array([-1.5, -0.5, -0.18, -0.04, 0.04, 0.18, 0.5, 1.5])
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for pred_arr, true_arr, c, lab, chance in [
        (rec["trial_pred_resp"], rec["trial_true_resp"],
         COLOR_RESP, "Response (3-class)", 1 / 3),
        (rec["trial_pred_stim"], rec["trial_true_stim"],
         COLOR_STIM, "Stim contrast (16-class)", 1 / 16),
        (rec["trial_pred_side"], rec["trial_true_side"],
         COLOR_SIDE, "Stimulus side (3-class)", 1 / 3),
    ]:
        eq = (pred_arr == true_arr).astype(np.float32)
        means, lo, hi = [], [], []
        for i in range(len(bins) - 1):
            mask = (diff >= bins[i]) & (diff < bins[i + 1])
            if mask.sum() < 5:
                means.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                continue
            v = eq[mask]
            m = v.mean()
            # Wilson-ish CI via bootstrap
            rng = np.random.default_rng(0)
            boot = np.array([
                rng.choice(v, size=len(v), replace=True).mean()
                for _ in range(500)
            ])
            means.append(m)
            lo.append(np.percentile(boot, 2.5))
            hi.append(np.percentile(boot, 97.5))
        means = np.array(means); lo = np.array(lo); hi = np.array(hi)
        ax.errorbar(
            centers, means, yerr=[means - lo, hi - means],
            color=c, marker="o", markersize=6, linewidth=1.6,
            capsize=3, label=lab,
        )
        ax.axhline(chance, color=c, linestyle=":", linewidth=0.6, alpha=0.4)
    ax.axvline(0, color="0.3", linestyle="--", linewidth=0.7)
    ax.set_xlabel("Signed contrast difference (right $-$ left)")
    ax.set_ylabel("Decoder trial-vote accuracy")
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(0, 1.0)
    ax.set_title(
        "C — Decoder accuracy vs trial difficulty\n"
        f"Steinmetz 39 sessions, {len(diff):,} held-out trials",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.savefig(OUT / "behav_C_psychometric.png", dpi=160, bbox_inches="tight")
    fig.savefig(OUT / "behav_C_psychometric.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote behav_C_psychometric.{png,pdf}")


# ---------------------------------------------------------------------------
# Mockup D — trial-aligned population-rate heatmap, sorted by behavior
# ---------------------------------------------------------------------------
def mockup_D(rec):
    pop = rec["aligned_pop_rate"]                 # (n_trials, WIN_LEN)
    true_resp = rec["aligned_resp_true"]          # 0=left, 1=no-go, 2=right
    # Sort: left-choice block, then no-go, then right-choice; within each
    # block sort by stim onset row activity (peak in window).
    order = np.argsort(true_resp, kind="stable")
    pop_sorted = pop[order]
    resp_sorted = true_resp[order]

    # Group boundaries
    boundaries = []
    for cls in (0, 1, 2):
        idx = np.where(resp_sorted == cls)[0]
        if len(idx):
            boundaries.append((cls, idx[0], idx[-1] + 1))

    t = (np.arange(WIN_LEN) - WIN_PRE_BINS) * BIN_S
    fig, ax = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)
    # robust color limits to keep visible
    vmax = np.percentile(pop_sorted, 98)
    im = ax.imshow(
        pop_sorted, aspect="auto", interpolation="nearest",
        extent=[t[0], t[-1], pop_sorted.shape[0], 0],
        cmap="magma", vmin=0, vmax=vmax,
    )
    ax.axvline(0, color="white", linestyle="--", linewidth=0.9, alpha=0.85)
    # Group separators
    cls_labels = {0: "left choice", 1: "no-go", 2: "right choice"}
    for cls, lo_idx, hi_idx in boundaries:
        ax.axhline(hi_idx, color="white", linestyle="-", linewidth=0.6,
                   alpha=0.6)
        ax.text(
            t[-1] + 0.04, (lo_idx + hi_idx) / 2,
            f"{cls_labels[cls]}\n(n={hi_idx - lo_idx})",
            va="center", ha="left", fontsize=8,
            color={0: COLOR_RESP, 1: "0.3", 2: COLOR_SIDE}[cls],
        )
    ax.set_xlabel("Time relative to stimulus onset (s)")
    ax.set_ylabel("Trial (sorted by response choice)")
    ax.set_title(
        "D — Trial-aligned predicted population rate, sorted by behavior\n"
        f"Steinmetz 39, {pop.shape[0]:,} held-out trials | "
        "Mamba seed 42 $\\sum_{\\text{neurons}}$ predicted rate",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, fraction=0.04,
                 label="$\\sum_{\\text{neurons}}$ predicted rate (spikes / 50 ms)")
    fig.savefig(OUT / "behav_D_pop_heatmap.png", dpi=160, bbox_inches="tight")
    fig.savefig(OUT / "behav_D_pop_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote behav_D_pop_heatmap.{png,pdf}")


# ---------------------------------------------------------------------------
# Mockup E — brain-region-resolved decoding
# ---------------------------------------------------------------------------
def mockup_E(rec):
    rr = rec["region_records"]
    if not rr:
        print("  no region records — skipping E")
        return
    # Aggregate by region (weighted by neuron count, across sessions)
    by_region = {}
    for r in rr:
        d = by_region.setdefault(r["region"], {
            "resp": [], "stim": [], "side": [], "n_neurons": [], "n_sess": 0,
        })
        d["resp"].append(r["resp"]); d["stim"].append(r["stim"])
        d["side"].append(r["side"]); d["n_neurons"].append(r["n_neurons"])
        d["n_sess"] += 1
    summary = []
    for region, d in by_region.items():
        if d["n_sess"] < 3:
            continue
        w = np.array(d["n_neurons"], dtype=float)
        summary.append({
            "region": region,
            "n_sess": d["n_sess"],
            "n_neurons_total": int(w.sum()),
            "resp": float(np.average(d["resp"], weights=w)),
            "stim": float(np.average(d["stim"], weights=w)),
            "side": float(np.average(d["side"], weights=w)),
        })
    # Top 12 regions by total neurons
    summary.sort(key=lambda x: -x["n_neurons_total"])
    top = summary[:12]
    # Sort within top by response decoding accuracy
    top.sort(key=lambda x: -x["resp"])

    regions = [t["region"] for t in top]
    resp = [t["resp"] for t in top]
    stim = [t["stim"] for t in top]
    side = [t["side"] for t in top]
    n_n = [t["n_neurons_total"] for t in top]

    x = np.arange(len(regions))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 4.4), constrained_layout=True)
    ax.bar(x - w, resp, width=w, color=COLOR_RESP, label="Response (3)",
           edgecolor="0.3", linewidth=0.4)
    ax.bar(x, stim, width=w, color=COLOR_STIM, label="Stim 16-class",
           edgecolor="0.3", linewidth=0.4)
    ax.bar(x + w, side, width=w, color=COLOR_SIDE, label="Side (3)",
           edgecolor="0.3", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r}\n($n_N$={n})" for r, n in zip(regions, n_n)],
        fontsize=8,
    )
    ax.axhline(1 / 3, color=COLOR_RESP, linestyle=":", linewidth=0.5, alpha=0.4)
    ax.axhline(1 / 16, color=COLOR_STIM, linestyle=":", linewidth=0.5,
               alpha=0.4)
    ax.set_ylabel("Decoder bin-level accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "E — Per-region decoding accuracy (input restricted to one Allen "
        "CCF region)\n"
        f"Steinmetz 39, top 12 regions by neuron count, weighted across "
        "sessions",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.savefig(OUT / "behav_E_region_decoding.png", dpi=160, bbox_inches="tight")
    fig.savefig(OUT / "behav_E_region_decoding.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote behav_E_region_decoding.{png,pdf}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print("Building trial records (training per-session decoders)...")
    rec = build_trial_records()
    print(f"  trials assembled: {len(rec['trial_pred_resp']):,}")
    print(f"  trial-aligned windows: {rec['aligned_resp_pred'].shape}")
    print(f"  region records: {len(rec['region_records'])}")
    print()
    print("Rendering mockups...")
    mockup_A(rec)
    mockup_B(rec)
    mockup_C(rec)
    mockup_D(rec)
    mockup_E(rec)
    print(f"\nDone. Figures in {OUT}/")


if __name__ == "__main__":
    main()
