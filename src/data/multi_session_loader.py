"""
Multi-session NWB data loader for training across multiple recording sessions.

Loads multiple NWB files, bins each independently, then zero-pads all sessions
to a common channel count (M_max) and concatenates along the time axis.  A
binary channel mask is maintained per session so the loss function can ignore
padded (non-existent) channels.

Key design choices:
    - Full data resolution: no unit subsampling.
    - Zero-pad to M_max: preserves all neurons; padding has no gradient signal
      because zero input contributes nothing to the LSTM hidden state, and the
      masked loss ignores padded output channels.
    - Session boundary gaps: ``history_bins`` columns of zeros are inserted
      between sessions to prevent temporal leakage across session boundaries.

Usage:
    from src.data.multi_session_loader import load_multi_session_nwb

    spike_counts, session_masks, metadata = load_multi_session_nwb(config)
"""

import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.binning import bin_spike_trains
from src.data.real_data_loader import load_nwb_spikes
from src.data.history_features import compute_history_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: pad a count matrix to a target number of channels
# ---------------------------------------------------------------------------

def pad_to_channels(
    spike_counts: np.ndarray,
    target_m: int,
) -> np.ndarray:
    """
    Zero-pad a spike-count matrix to a target number of channels.

    If the matrix already has target_m channels, returns it unchanged.

    Args:
        spike_counts: Shape (M, T), spike count matrix.
        target_m: Target number of channels (must be >= M).

    Returns:
        Zero-padded matrix of shape (target_m, T).
    """
    m, t = spike_counts.shape
    if m == target_m:
        return spike_counts
    if m > target_m:
        raise ValueError(
            f"Cannot pad: source has {m} channels but target is {target_m}"
        )
    # Zero-pad along channel axis
    padded = np.zeros((target_m, t), dtype=spike_counts.dtype)
    padded[:m, :] = spike_counts
    return padded


def build_channel_mask(
    num_real_channels: int,
    total_channels: int,
) -> np.ndarray:
    """
    Build a binary channel mask: 1 for real channels, 0 for padding.

    Args:
        num_real_channels: Number of real (non-padded) channels.
        total_channels: Total number of channels after padding (M_max).

    Returns:
        Binary mask of shape (total_channels,), dtype float32.
    """
    mask = np.zeros(total_channels, dtype=np.float32)
    mask[:num_real_channels] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Main multi-session loader
# ---------------------------------------------------------------------------

def load_multi_session_nwb(
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load and concatenate multiple NWB sessions into a single count matrix.

    For each NWB file:
        1. Load spike trains via ``load_nwb_spikes()``
        2. Bin via ``bin_spike_trains()``
        3. Record the number of real channels

    Then:
        4. Determine M_max across all sessions
        5. Zero-pad each session to (M_max, T_i)
        6. Insert ``history_bins``-wide zero gaps between sessions
        7. Concatenate → (M_max, T_total)
        8. Build per-bin mask index mapping each time bin to its session's
           channel mask

    Args:
        config: Data configuration dictionary. Expected keys:
            - source.type: Must be "nwb_multi"
            - source.glob: Glob pattern for NWB files (e.g., "data/raw/*.nwb")
            - source.file_list: Optional list of specific file paths
              (overrides glob if set)
            - All other keys forwarded to ``load_nwb_spikes()`` per session

    Returns:
        Tuple of:
            - spike_counts: np.ndarray, shape (M_max, T_total), dtype int32
            - mask_index: np.ndarray, shape (T_total,), dtype int32 — maps
              each bin to a row in ``session_masks``
            - metadata: dict with:
                - session_masks: np.ndarray (num_sessions, M_max), float32
                - m_max: int
                - num_sessions: int
                - session_details: list of per-session metadata dicts
                - total_bins: int
                - gap_bins: int (history_bins used for boundaries)
    """
    source_config = config.get("source", {})
    history_bins = config.get("history_bins", 50)
    bin_width_ms = config.get("bin_width_ms", 10.0)

    # --- Discover NWB files ---
    file_list = source_config.get("file_list", None)
    if file_list:
        # Explicit file list takes priority
        nwb_paths = [Path(p) for p in file_list]
    else:
        # Glob pattern
        glob_pattern = source_config.get("glob", "data/raw/Steinmetz2019_*.nwb")
        # If glob is relative, resolve from project root
        glob_path = Path(glob_pattern)
        if not glob_path.is_absolute():
            nwb_paths = sorted(Path(".").glob(glob_pattern))
        else:
            nwb_paths = sorted(glob_path.parent.glob(glob_path.name))

    if not nwb_paths:
        raise FileNotFoundError(
            f"No NWB files found matching config: {source_config}"
        )

    logger.info(
        "Multi-session loader: found %d NWB files", len(nwb_paths),
    )

    # --- Load and bin each session ---
    session_counts: List[np.ndarray] = []  # Each (M_i, T_i)
    session_real_m: List[int] = []          # Real channel count per session
    session_details: List[Dict[str, Any]] = []

    for i, nwb_path in enumerate(nwb_paths):
        logger.info(
            "[%d/%d] Loading session: %s",
            i + 1, len(nwb_paths), nwb_path.name,
        )

        # Build per-session config (override path, keep all other settings)
        session_config = {**config, "source": {**source_config, "path": str(nwb_path)}}

        # Load spike trains
        sorting, load_meta = load_nwb_spikes(session_config)

        # Bin spike trains
        counts, bin_meta = bin_spike_trains(
            sorting, bin_width_ms=bin_width_ms,
        )

        m_i, t_i = counts.shape
        logger.info(
            "  Session %d: %d units, %d bins (%.1fs)",
            i, m_i, t_i, t_i * bin_width_ms / 1000,
        )

        session_counts.append(counts)
        session_real_m.append(m_i)
        session_details.append({
            "file": str(nwb_path),
            "num_units": m_i,
            "num_bins": t_i,
            "duration_s": t_i * bin_width_ms / 1000,
            "load_metadata": load_meta,
            "bin_metadata": bin_meta,
        })

    # --- Determine M_max ---
    m_max = max(session_real_m)
    logger.info(
        "M_max = %d (range: %d to %d across %d sessions)",
        m_max, min(session_real_m), m_max, len(session_counts),
    )

    # --- Build per-session channel masks ---
    num_sessions = len(session_counts)
    session_masks = np.zeros((num_sessions, m_max), dtype=np.float32)
    for i, real_m in enumerate(session_real_m):
        session_masks[i] = build_channel_mask(real_m, m_max)

    # --- Zero-pad and concatenate with boundary gaps ---
    padded_segments: List[np.ndarray] = []  # Padded count segments
    mask_segments: List[np.ndarray] = []     # Mask index per bin

    for i, counts in enumerate(session_counts):
        # Pad to M_max
        padded = pad_to_channels(counts, m_max)
        t_i = padded.shape[1]

        # Append padded session
        padded_segments.append(padded)
        mask_segments.append(np.full(t_i, i, dtype=np.int32))

        # Insert boundary gap between sessions (not after the last one)
        if i < num_sessions - 1:
            gap = np.zeros((m_max, history_bins), dtype=padded.dtype)
            padded_segments.append(gap)
            # Gap bins use session -1 (will be handled specially — samples
            # spanning the gap are naturally invalid since they cross the
            # zero-padded boundary)
            mask_segments.append(np.full(history_bins, -1, dtype=np.int32))

    # Concatenate along time axis
    spike_counts = np.concatenate(padded_segments, axis=1)
    mask_index = np.concatenate(mask_segments)

    t_total = spike_counts.shape[1]
    total_gap_bins = history_bins * (num_sessions - 1) if num_sessions > 1 else 0
    total_data_bins = t_total - total_gap_bins

    logger.info(
        "Concatenated: shape (%d, %d) — %d data bins + %d gap bins (%d sessions)",
        m_max, t_total, total_data_bins, total_gap_bins, num_sessions,
    )

    # --- Build metadata ---
    metadata = {
        "source": "nwb_multi",
        "num_sessions": num_sessions,
        "m_max": m_max,
        "session_real_m": session_real_m,
        "session_masks": session_masks,
        "total_bins": t_total,
        "total_data_bins": total_data_bins,
        "gap_bins": history_bins,
        "total_gap_bins": total_gap_bins,
        "session_details": session_details,
    }

    return spike_counts, mask_index, metadata


# ---------------------------------------------------------------------------
# Masked dataset: returns (x, y, mask) triples
# ---------------------------------------------------------------------------

class MaskedSpikeCountDataset(Dataset):
    """
    PyTorch Dataset for multi-session spike-count forecasting with channel masks.

    Extends the standard sliding-window approach to return a per-sample channel
    mask alongside the (input, target) pair.  The mask is used by the trainer
    to compute loss only on real (non-padded) channels.

    Supports optional history features: when features are provided, the input
    data panel has shape (input_size, T) where input_size = M_max + n_feat * M_max,
    but targets always come from the first output_channels (= M_max) channels.

    Supports optional covariates: when covariates are provided (shape
    (n_covariates, T_total)), a 4th tensor is returned with covariates.
    The covariate shape depends on ``covariate_mode``:
        - ``"additive"``: returns target-bin covariates (n_covariates,)
        - ``"temporal"``: returns full-window covariates (T, n_covariates)
    Without covariates, only the standard 3-tuple is returned.

    Yields (x, y, mask) or (x, y, mask, covariates) tuples:
        - x: (T, input_size) float tensor — history window (may include features)
        - y: (output_channels,) float tensor — next-step target (spike counts only)
        - mask: (output_channels,) float tensor — 1 for real channels, 0 for padding
        - covariates: covariate tensor (shape depends on covariate_mode, optional)

    Args:
        spike_counts: Shape (input_size, T_total), padded count matrix
            (optionally with concatenated history features along axis 0).
        mask_index: Shape (T_total,), maps each bin to a session index (or -1 for gaps).
        session_masks: Shape (num_sessions, output_channels), binary channel masks.
        history_bins: Number of history bins per sample.
        dtype: Torch dtype for output tensors.
        output_channels: Number of output channels for targets (default: all).
            When history features are appended, this should be M_max (the raw
            channel count before feature expansion).
        covariates: Optional covariate matrix, shape (n_covariates, T_total).
            When provided, __getitem__ returns a 4-tuple including covariates.
        covariate_mode: How covariates are returned. ``"additive"`` returns
            the target-bin covariate vector (n_cov,). ``"temporal"`` returns
            the full history-window covariates (T, n_cov). Default: ``"additive"``.
    """

    def __init__(
        self,
        spike_counts: np.ndarray,
        mask_index: np.ndarray,
        session_masks: np.ndarray,
        history_bins: int = 50,
        dtype: torch.dtype = torch.float32,
        output_channels: Optional[int] = None,
        covariates: Optional[np.ndarray] = None,
        covariate_mode: str = "additive",
        mask_self_history: bool = False,
    ):
        super().__init__()

        if spike_counts.ndim != 2:
            raise ValueError(
                f"spike_counts must be 2D (M, T_total), got {spike_counts.shape}"
            )

        self.input_size, self.total_bins = spike_counts.shape
        # output_channels defaults to full input width (no features case)
        self.output_channels = output_channels or self.input_size
        self.m_max = self.output_channels  # For backward compat
        self.history_bins = history_bins
        self.dtype = dtype

        # Auto-history ablation: when True, zero out each neuron's own
        # column in the input window so the model must predict from
        # population activity alone (Tier 1D — KOSMOS recommendation).
        self.mask_self_history = mask_self_history

        # Convert data to tensor (time-first for slicing)
        # Shape: (T_total, input_size)
        self._data = torch.tensor(spike_counts.T, dtype=dtype)

        # Store mask_index and session_masks
        self._mask_index = mask_index  # (T_total,)

        # Optional covariates: (n_covariates, T_total) -> (T_total, n_covariates)
        if covariates is not None:
            self._covariates = torch.tensor(covariates.T, dtype=dtype)
        else:
            self._covariates = None

        # Covariate mode: "additive" returns target-bin (n_cov,),
        # "temporal" returns full window (T, n_cov)
        if covariate_mode not in ("additive", "temporal"):
            raise ValueError(
                f"covariate_mode must be 'additive' or 'temporal', "
                f"got '{covariate_mode}'"
            )
        self.covariate_mode = covariate_mode
        self._session_masks = torch.tensor(session_masks, dtype=dtype)  # (S, output_channels)

        # Precompute valid sample indices: a sample at index t uses
        # bins [t, t + history_bins) for input and bin t + history_bins as target.
        # A sample is valid only if ALL bins in the window AND the target bin
        # belong to the same session (no gaps, no cross-session leakage).
        self._valid_indices = self._compute_valid_indices()

        logger.info(
            "MaskedSpikeCountDataset: input_size=%d, output_channels=%d, "
            "T_total=%d, history=%d, valid_samples=%d (of %d possible)%s",
            self.input_size, self.output_channels,
            self.total_bins, history_bins,
            len(self._valid_indices),
            self.total_bins - history_bins,
            ", mask_self_history=True" if mask_self_history else "",
        )

    def _compute_valid_indices(self) -> np.ndarray:
        """
        Find sample indices where the full window + target stays within
        one session (no gap bins, no cross-session boundary).

        Uses a vectorized approach: a sample at index t is valid iff all
        bins [t, t + history_bins] belong to the same non-negative session.
        We detect boundaries via np.diff and compute a "distance to nearest
        boundary" array, then select indices where the distance exceeds
        history_bins.

        Returns:
            Array of valid starting indices.
        """
        mi = self._mask_index  # (T_total,)
        T = len(mi)
        n_samples = T - self.history_bins

        if n_samples <= 0:
            return np.array([], dtype=np.int64)

        # Fast path: if all bins belong to the same non-negative session
        # (single-session lazy loading), every index is valid.
        if np.all(mi == mi[0]) and mi[0] >= 0:
            return np.arange(n_samples, dtype=np.int64)

        # General case: find positions where session changes or is a gap
        # A "boundary" is where mi[t] != mi[t-1] or mi[t] < 0
        is_gap = mi < 0  # (T,)

        # Session-change positions: where consecutive bins differ
        changes = np.empty(T, dtype=bool)
        changes[0] = False  # First bin has no predecessor
        changes[1:] = mi[1:] != mi[:-1]

        # A bin is "bad" if it's a gap or a session change point
        is_bad = is_gap | changes  # (T,)

        # Compute distance from each bin to the nearest preceding bad bin.
        # At a bad bin, distance = 0. Otherwise, distance increments by 1.
        # We need distance > history_bins for a valid sample start.
        dist_from_boundary = np.zeros(T, dtype=np.int64)
        for t in range(1, T):
            if is_bad[t]:
                dist_from_boundary[t] = 0
            else:
                dist_from_boundary[t] = dist_from_boundary[t - 1] + 1

        # A sample at index t uses bins [t, t + history_bins].
        # The TARGET bin is at t + history_bins.
        # All bins in the window must have dist_from_boundary >= history_bins
        # (meaning they are at least history_bins bins away from any boundary).
        # It's sufficient to check that the target bin at t + history_bins has
        # dist_from_boundary >= history_bins (since dist is monotonically
        # increasing within a session segment).
        target_dist = dist_from_boundary[self.history_bins:]  # (n_samples,)
        valid_mask = target_dist >= self.history_bins
        # Also ensure the target bin is from a real session (not a gap)
        target_sessions = mi[self.history_bins:]  # (n_samples,)
        valid_mask &= target_sessions >= 0

        return np.where(valid_mask)[0].astype(np.int64)

    def __len__(self) -> int:
        """Return number of valid samples."""
        return len(self._valid_indices)

    def __getitem__(self, idx: int):
        """
        Get (input, target, mask[, covariates]) tuple at index idx.

        Returns a 3-tuple (x, y, mask) when no covariates are configured,
        or a 4-tuple (x, y, mask, cov) when covariates are present.

        Covariate shape depends on ``covariate_mode``:
            - ``"additive"``: (n_covariates,) — target-bin only
            - ``"temporal"``: (T, n_covariates) — full history window

        Args:
            idx: Sample index in [0, num_valid_samples).

        Returns:
            Tuple of:
                - x: (T, input_size) float tensor — history window
                      (includes features if present)
                - y: (output_channels,) float tensor — next-step target
                      (spike counts only, no features)
                - mask: (output_channels,) float tensor — channel mask
                - covariates: covariate tensor (only when covariates present)
        """
        if idx < 0 or idx >= len(self._valid_indices):
            raise IndexError(
                f"Index {idx} out of range [0, {len(self._valid_indices)})"
            )

        t = self._valid_indices[idx]

        # Input window: bins [t, t + history_bins), all channels + features
        x = self._data[t : t + self.history_bins, :]

        # Auto-history ablation: zero ALL spike-count columns in the input.
        #
        # WHY THIS ZEROS EVERYTHING (not a bug):
        # The model predicts all M neurons simultaneously. For neuron j's
        # self-history to be absent, column j must be zero. Since ALL
        # neurons need their self-history removed at the same time, ALL
        # columns must be zero. This is the mathematically correct
        # "simultaneous diagonal masking" for shared prediction.
        #
        # This is a HARSH ablation: r≈0 is expected because no neuron
        # has access to any spike-count history (its own or others').
        #
        # For the more informative PER-NEURON ablation (zero only column j,
        # measure neuron j's drop, repeat for each j independently),
        # use: scripts/eval_autohistory_ablation.py
        if self.mask_self_history:
            x = x.clone()
            x[:, :self.output_channels] = 0.0

        # Target: bin at t + history_bins, spike counts only (first output_channels)
        y = self._data[t + self.history_bins, :self.output_channels]

        # Channel mask: from the session that owns the target bin
        session_idx = self._mask_index[t + self.history_bins]
        mask = self._session_masks[session_idx]

        # Return 4-tuple when covariates are available, 3-tuple otherwise
        if self._covariates is not None:
            if self.covariate_mode == "temporal":
                # Full history window covariates: (T, n_covariates)
                cov = self._covariates[t : t + self.history_bins, :]
            else:
                # Target-bin only: (n_covariates,)
                cov = self._covariates[t + self.history_bins, :]
            return x, y, mask, cov

        return x, y, mask


# ---------------------------------------------------------------------------
# DataLoader creation for masked multi-session data
# ---------------------------------------------------------------------------

def create_masked_dataloaders(
    spike_counts: np.ndarray,
    mask_index: np.ndarray,
    session_masks: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders for multi-session masked data.

    Performs temporal splitting on the concatenated matrix and builds
    MaskedSpikeCountDataset for each split.

    Args:
        spike_counts: Shape (M_max, T_total), padded count matrix.
        mask_index: Shape (T_total,), session index per bin.
        session_masks: Shape (num_sessions, M_max), binary masks.
        config: Data configuration dictionary.

    Returns:
        Dict with keys 'train', 'val', 'test', each a DataLoader.
    """
    # Extract config values
    history_bins = config.get("history_bins", 50)
    splits = config.get("splits", {"train": 0.7, "val": 0.15, "test": 0.15})
    batch_size = config.get("batch_size", 128)
    compute = config.get("compute", {})
    num_workers = compute.get("num_workers", 0)
    pin_memory = compute.get("pin_memory", False)

    # Temporal split boundaries
    t_total = spike_counts.shape[1]
    train_end = int(t_total * splits["train"])
    val_end = train_end + int(t_total * splits["val"])

    # Split counts and mask_index along time
    splits_data = {
        "train": (spike_counts[:, :train_end], mask_index[:train_end]),
        "val": (spike_counts[:, train_end:val_end], mask_index[train_end:val_end]),
        "test": (spike_counts[:, val_end:], mask_index[val_end:]),
    }

    logger.info(
        "Temporal split: train=%d bins, val=%d bins, test=%d bins",
        train_end,
        val_end - train_end,
        t_total - val_end,
    )

    # Build datasets and loaders
    loaders = {}
    for split_name, (counts_split, mask_split) in splits_data.items():
        ds = MaskedSpikeCountDataset(
            spike_counts=counts_split,
            mask_index=mask_split,
            session_masks=session_masks,
            history_bins=history_bins,
        )

        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split_name == "train"),
        )

        logger.info(
            "  %s: %d valid samples, %d batches",
            split_name, len(ds), len(loaders[split_name]),
        )

    return loaders


# ---------------------------------------------------------------------------
# Lazy per-session loading: preprocess + cache, then session-cycling loader
# ---------------------------------------------------------------------------

def preprocess_and_cache(
    config: Dict[str, Any],
    cache_dir: Optional[str] = None,
    force_reprocess: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Preprocess all NWB sessions: load, bin, and save as cached .npy files.

    This is the first pass of the two-pass lazy loading architecture.
    Each session is processed one at a time to keep memory usage low.
    Count matrices are stored as uint8 (values clamped at 255 with a
    warning if overflow occurs).

    The cache is reused on subsequent runs unless ``force_reprocess=True``.

    Args:
        config: Data configuration dictionary (same as load_multi_session_nwb).
        cache_dir: Optional override for cache directory path.
            Default: ``data/processed/session_cache/``.
        force_reprocess: If True, reprocess even if cache exists.

    Returns:
        Tuple of:
            - cache_path: Path to the cache directory
            - metadata: dict with m_max, session info, split boundaries
    """
    import json

    source_config = config.get("source", {})
    bin_width_ms = config.get("bin_width_ms", 10.0)
    history_bins = config.get("history_bins", 50)
    splits = config.get("splits", {"train": 0.7, "val": 0.15, "test": 0.15})

    # Check if history features are enabled in this config
    hf_cfg = config.get("history_features", {})
    hf_enabled = hf_cfg.get("enabled", False)

    # Check if covariates are enabled
    cov_cfg = config.get("covariates", {})
    cov_enabled = cov_cfg.get("enabled", False)

    # --- Determine cache directory ---
    if cache_dir:
        cache_path = Path(cache_dir)
    else:
        cache_path = Path("data/processed/session_cache")

    metadata_file = cache_path / "metadata.json"

    # --- Check if cache already exists ---
    if metadata_file.exists() and not force_reprocess:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        logger.info(
            "Cache found at %s with %d sessions (M_max=%d). Skipping preprocessing.",
            cache_path, metadata["num_sessions"], metadata["m_max"],
        )
        return cache_path, metadata

    # --- Discover NWB files ---
    file_list = source_config.get("file_list", None)
    if file_list:
        nwb_paths = [Path(p) for p in file_list]
    else:
        glob_pattern = source_config.get("glob", "data/raw/Steinmetz2019_*.nwb")
        glob_path = Path(glob_pattern)
        if not glob_path.is_absolute():
            nwb_paths = sorted(Path(".").glob(glob_pattern))
        else:
            nwb_paths = sorted(glob_path.parent.glob(glob_path.name))

    if not nwb_paths:
        raise FileNotFoundError(
            f"No NWB files found matching config: {source_config}"
        )

    logger.info(
        "Preprocessing %d NWB files → cache at %s", len(nwb_paths), cache_path,
    )

    # Create cache directory
    cache_path.mkdir(parents=True, exist_ok=True)

    # --- Process each session ---
    session_info: List[Dict[str, Any]] = []
    overflow_total = 0

    for i, nwb_path in enumerate(nwb_paths):
        logger.info(
            "[%d/%d] Processing: %s",
            i + 1, len(nwb_paths), nwb_path.name,
        )

        # Build per-session config
        session_config = {
            **config,
            "source": {**source_config, "path": str(nwb_path)},
        }

        # Load spike trains
        sorting, load_meta = load_nwb_spikes(session_config)

        # Bin spike trains
        counts, bin_meta = bin_spike_trains(sorting, bin_width_ms=bin_width_ms)
        m_i, t_i = counts.shape

        # Check for uint8 overflow (any count > 255)
        overflow_count = int(np.sum(counts > 255))
        if overflow_count > 0:
            overflow_total += overflow_count
            max_val = int(counts.max())
            logger.warning(
                "  Session %d (%s): %d bins exceed uint8 range (max=%d). "
                "Clamping to 255.",
                i, nwb_path.name, overflow_count, max_val,
            )

        # Save as uint8 (clamp at 255)
        counts_u8 = np.clip(counts, 0, 255).astype(np.uint8)
        npy_path = cache_path / f"session_{i:03d}.npy"
        np.save(npy_path, counts_u8)

        # Compute and cache history features if enabled (ADR-0009)
        n_features_per_channel = 0
        if hf_enabled:
            features, n_feat = compute_history_features(counts, config)
            n_features_per_channel = n_feat
            if n_feat > 0:
                # Save features as float32 (features are already normalized)
                feat_path = cache_path / f"session_{i:03d}_features.npy"
                np.save(feat_path, features.astype(np.float32))
                logger.info(
                    "  Session %d: saved %d history features (%d rows)",
                    i, n_feat, features.shape[0],
                )

        # Extract and cache covariates if enabled (ADR-0012/0013)
        n_covariates = 0
        covariate_names = []
        if cov_enabled:
            from src.data.nwb_covariates import extract_stimulus_features
            cov_features = cov_cfg.get("features", None)
            covariates, cov_names = extract_stimulus_features(
                nwb_path=nwb_path,
                num_bins=t_i,
                bin_width_ms=bin_width_ms,
                feature_list=cov_features,
            )
            covariate_names = list(cov_names)

            # --- LFP band-power covariates (ADR-0015 Tier A+) ---
            lfp_cfg = cov_cfg.get("lfp", {})
            if lfp_cfg.get("enabled", False):
                from src.data.lfp_nwb_reader import load_lfp_from_nwb
                from src.data.lfp_features import (
                    compute_lfp_band_power, DEFAULT_BANDS,
                )

                # Load LFP from this session's NWB file
                lfp_result = load_lfp_from_nwb(nwb_path)
                if lfp_result is not None:
                    lfp_signal, lfp_fs = lfp_result

                    # Limit to max_channels to keep covariates manageable
                    max_ch = lfp_cfg.get("max_channels", 1)
                    if lfp_signal.shape[0] > max_ch:
                        lfp_signal = lfp_signal[:max_ch, :]

                    # Build bin edges in seconds for alignment
                    bin_edges_s = np.arange(t_i + 1) * (bin_width_ms / 1000.0)

                    # Get band config (use defaults if not specified)
                    bands_cfg = lfp_cfg.get("bands", None)
                    if bands_cfg is not None:
                        # Convert list values [lo, hi] to tuples
                        bands = {
                            k: tuple(v) for k, v in bands_cfg.items()
                        }
                    else:
                        bands = DEFAULT_BANDS

                    # Compute band power: (n_bands * n_ch, n_bins)
                    lfp_power = compute_lfp_band_power(
                        lfp_signal, lfp_fs, bin_edges_s, bands=bands,
                    )
                    n_lfp_features = lfp_power.shape[0]

                    # Build LFP covariate names
                    n_lfp_ch = lfp_signal.shape[0]
                    lfp_names = []
                    for ch in range(n_lfp_ch):
                        for band_name in bands.keys():
                            lfp_names.append(f"lfp_ch{ch}_{band_name}")

                    # Concatenate with stimulus covariates
                    covariates = np.concatenate(
                        [covariates, lfp_power], axis=0,
                    )
                    covariate_names.extend(lfp_names)
                    logger.info(
                        "  Session %d: LFP added %d features (%d ch × %d bands)",
                        i, n_lfp_features, n_lfp_ch, len(bands),
                    )
                    del lfp_signal, lfp_power
                else:
                    # No LFP in this NWB file — zero-fill to match expected shape
                    # Use n_lfp_expected from first session that had LFP, or
                    # default to max_channels * n_bands
                    max_ch = lfp_cfg.get("max_channels", 1)
                    bands_cfg = lfp_cfg.get("bands", None)
                    n_bands = len(bands_cfg) if bands_cfg else len(DEFAULT_BANDS)
                    n_lfp_features = max_ch * n_bands
                    lfp_zeros = np.zeros(
                        (n_lfp_features, t_i), dtype=np.float32,
                    )
                    covariates = np.concatenate(
                        [covariates, lfp_zeros], axis=0,
                    )
                    # Add placeholder names
                    for ch in range(max_ch):
                        band_names = list(bands_cfg.keys()) if bands_cfg else list(DEFAULT_BANDS.keys())
                        for band_name in band_names:
                            covariate_names.append(f"lfp_ch{ch}_{band_name}")
                    logger.info(
                        "  Session %d: no LFP in NWB — zero-filled %d features",
                        i, n_lfp_features,
                    )

            n_covariates = covariates.shape[0]
            cov_path = cache_path / f"session_{i:03d}_covariates.npy"
            np.save(cov_path, covariates.astype(np.float32))
            logger.info(
                "  Session %d: saved %d covariates (%s) → %s",
                i, n_covariates, ", ".join(covariate_names), cov_path.name,
            )

        # Compute per-session split boundaries
        train_end = int(t_i * splits["train"])
        val_end = train_end + int(t_i * splits["val"])

        session_info.append({
            "index": i,
            "file": str(nwb_path),
            "npy_file": str(npy_path),
            "num_units": m_i,
            "num_bins": t_i,
            "duration_s": round(t_i * bin_width_ms / 1000, 2),
            "split_boundaries": {
                "train_end": train_end,
                "val_end": val_end,
            },
            # Per-unit brain region labels (list of strings, or None)
            "brain_regions": load_meta.get("brain_regions", None),
        })

        logger.info(
            "  Session %d: %d units, %d bins (%.1fs) → %s",
            i, m_i, t_i, t_i * bin_width_ms / 1000, npy_path.name,
        )

    # --- Compute M_max across all sessions ---
    all_m = [s["num_units"] for s in session_info]
    m_max = max(all_m)

    logger.info(
        "M_max = %d (range: %d to %d across %d sessions)",
        m_max, min(all_m), m_max, len(session_info),
    )

    if overflow_total > 0:
        logger.warning(
            "Total uint8 overflows across all sessions: %d (clamped to 255)",
            overflow_total,
        )

    # --- Write metadata ---
    metadata = {
        "num_sessions": len(session_info),
        "m_max": m_max,
        "history_bins": history_bins,
        "bin_width_ms": bin_width_ms,
        "n_features_per_channel": n_features_per_channel,
        "n_covariates": n_covariates,
        "covariate_names": covariate_names,
        "sessions": session_info,
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Cache complete: %d sessions, M_max=%d", len(session_info), m_max)

    return cache_path, metadata


class SessionCyclingLoader:
    """
    Lazy per-session DataLoader that loads one cached session at a time.

    Behaves like a standard DataLoader (iterable yielding batches), but
    only ever holds one session's data in memory.  Session order is
    shuffled each epoch for training, fixed for val/test.

    One full iteration visits every session exactly once, yielding all
    batches from each session before moving to the next.

    Args:
        cache_dir: Path to the cache directory with .npy files and metadata.
        metadata: Metadata dict from ``preprocess_and_cache()``.
        split: Which split to serve: ``"train"``, ``"val"``, or ``"test"``.
        config: Data configuration dict (for batch_size, etc).
        shuffle_sessions: Whether to shuffle session order each epoch
            (default True for train, False otherwise).
    """

    def __init__(
        self,
        cache_dir: Path,
        metadata: Dict[str, Any],
        split: str,
        config: Dict[str, Any],
        shuffle_sessions: Optional[bool] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.metadata = metadata
        self.split = split
        self.m_max = metadata["m_max"]
        self.history_bins = metadata.get("history_bins", config.get("history_bins", 50))

        # Covariate mode: "additive" or "temporal" (ADR-0012)
        cov_cfg = config.get("covariates", {})
        self.covariate_mode = cov_cfg.get("mode", "additive")

        # Auto-history ablation (Tier 1D — KOSMOS recommendation):
        # when enabled, each neuron's own spike history is zeroed in input,
        # forcing the model to predict from population activity alone.
        self.mask_self_history = config.get("mask_self_history", False)
        if self.mask_self_history:
            logger.info("Auto-history ablation ENABLED: self-history will be zeroed")

        # DataLoader config
        # NOTE: Force num_workers=0 on Windows only.  The lazy loader
        # creates a NEW DataLoader for each session (39+ per epoch).
        # On Windows, each DataLoader with num_workers > 0 consumes
        # shared file mappings that are not reclaimed fast enough,
        # causing error 1455 ("Couldn't open shared file mapping").
        # On Linux (e.g., NRP containers), multi-worker loading is safe
        # and keeps the GPU fed while the next batch is prepared.
        import platform
        self.batch_size = config.get("batch_size", 128)
        if platform.system() == "Windows":
            self.num_workers = 0
            self.pin_memory = False
        else:
            compute_cfg = config.get("compute", {})
            self.num_workers = compute_cfg.get("num_workers", 4)
            self.pin_memory = compute_cfg.get("pin_memory", True)

        # Shuffle sessions each epoch for training by default
        if shuffle_sessions is None:
            self.shuffle_sessions = (split == "train")
        else:
            self.shuffle_sessions = shuffle_sessions

        # Shuffle within-session samples for training
        self.shuffle_samples = (split == "train")

        # Precompute total estimated batches for __len__
        self._total_samples = 0
        for sess in metadata["sessions"]:
            split_len = self._get_split_length(sess)
            # Estimate valid samples (subtract history_bins for sliding window)
            valid_approx = max(0, split_len - self.history_bins)
            self._total_samples += valid_approx

        self._total_batches = max(1, self._total_samples // self.batch_size)

        # Session-specific head support (Phase 1):
        # - current_session_id: set during iteration, read by trainer
        # - session_neuron_counts: maps session_id → N_i for building model
        self.current_session_id: Optional[str] = None
        self.session_neuron_counts: Dict[str, int] = {
            f"session_{s['index']:03d}": s["num_units"]
            for s in metadata["sessions"]
        }

    def _get_split_length(self, session_info: Dict[str, Any]) -> int:
        """Get the number of bins in the requested split for a session."""
        bounds = session_info["split_boundaries"]
        t_total = session_info["num_bins"]

        if self.split == "train":
            return bounds["train_end"]
        elif self.split == "val":
            return bounds["val_end"] - bounds["train_end"]
        else:  # test
            return t_total - bounds["val_end"]

    def _get_split_slice(
        self, session_info: Dict[str, Any],
    ) -> Tuple[int, int]:
        """Get (start, end) indices for the requested split of a session."""
        bounds = session_info["split_boundaries"]
        t_total = session_info["num_bins"]

        if self.split == "train":
            return 0, bounds["train_end"]
        elif self.split == "val":
            return bounds["train_end"], bounds["val_end"]
        else:  # test
            return bounds["val_end"], t_total

    def __len__(self) -> int:
        """Estimated total batches across all sessions."""
        return self._total_batches

    def __iter__(self):
        """
        Iterate over all sessions, yielding (x, y, mask[, cov]) batches.

        For each session:
          1. Load cached .npy from disk (counts + optional features)
          2. Slice to the requested split
          3. Pad counts and features to M_max
          4. Concatenate counts + features for expanded input
          5. Load and slice covariates if cached
          6. Build MaskedSpikeCountDataset (single-session)
          7. Yield all shuffled batches
          8. Delete session data to free RAM
        """
        # Check if history features were cached
        n_feat = self.metadata.get("n_features_per_channel", 0)
        # Check if covariates were cached
        n_cov = self.metadata.get("n_covariates", 0)

        # Determine session order
        session_indices = list(range(self.metadata["num_sessions"]))
        if self.shuffle_sessions:
            import random
            random.shuffle(session_indices)

        for sess_idx in session_indices:
            sess_info = self.metadata["sessions"][sess_idx]

            # 1. Load cached count matrix (uint8)
            npy_path = self.cache_dir / f"session_{sess_idx:03d}.npy"
            counts_u8 = np.load(npy_path)  # (M_i, T_i), uint8

            # 2. Slice to requested split
            start, end = self._get_split_slice(sess_info)
            split_len = end - start
            if split_len <= self.history_bins:
                # Skip sessions where the split is too short
                del counts_u8
                continue

            counts_split = counts_u8[:, start:end].astype(np.int32)
            del counts_u8  # Free the full session immediately

            # 3. Pad counts to M_max
            m_i = counts_split.shape[0]
            padded = pad_to_channels(counts_split, self.m_max)
            del counts_split

            # 3b. Load and pad history features if they exist
            if n_feat > 0:
                feat_path = self.cache_dir / f"session_{sess_idx:03d}_features.npy"
                if feat_path.exists():
                    features = np.load(feat_path)  # (n_feat * M_i, T_i), float32
                    features_split = features[:, start:end]
                    del features
                    # Pad features: n_feat blocks of M_i -> n_feat blocks of M_max
                    feat_padded = np.zeros(
                        (n_feat * self.m_max, features_split.shape[1]),
                        dtype=np.float32,
                    )
                    for f in range(n_feat):
                        feat_padded[f * self.m_max : f * self.m_max + m_i, :] = \
                            features_split[f * m_i : (f + 1) * m_i, :]
                    del features_split
                    # Concatenate: (M_max + n_feat * M_max, T_split)
                    padded = np.concatenate(
                        [padded.astype(np.float32), feat_padded], axis=0,
                    )
                    del feat_padded

            # 4b. Load and slice covariates if they exist
            cov_split = None
            if n_cov > 0:
                cov_path = self.cache_dir / f"session_{sess_idx:03d}_covariates.npy"
                if cov_path.exists():
                    cov_full = np.load(cov_path)  # (n_cov, T_i), float32
                    cov_split = cov_full[:, start:end]
                    del cov_full

            # 5. Build single-session mask and mask_index
            # All bins in this split belong to one session (session 0 in
            # the single-session dataset)
            mask = build_channel_mask(m_i, self.m_max)
            session_masks = mask.reshape(1, -1)  # (1, M_max)
            mask_index = np.zeros(padded.shape[1], dtype=np.int32)  # All 0s

            # 6. Create dataset for this session's split
            # output_channels = M_max (targets are always raw spike counts)
            ds = MaskedSpikeCountDataset(
                spike_counts=padded,
                mask_index=mask_index,
                session_masks=session_masks,
                history_bins=self.history_bins,
                output_channels=self.m_max,
                covariates=cov_split,
                covariate_mode=self.covariate_mode,
                mask_self_history=self.mask_self_history,
            )
            del padded, mask_index, cov_split

            if len(ds) == 0:
                del ds
                continue

            # 6. Create a DataLoader and yield batches
            loader = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=self.shuffle_samples,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=(self.split == "train"),
            )

            for batch in loader:
                # Expose current session identity for session-specific heads
                self.current_session_id = f"session_{sess_idx:03d}"
                yield batch

            # 7. Free memory before loading next session
            del loader, ds
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def create_lazy_dataloaders(
    cache_dir: Path,
    metadata: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, "SessionCyclingLoader"]:
    """
    Create train/val/test SessionCyclingLoaders for lazy multi-session training.

    This is the lazy counterpart to ``create_masked_dataloaders()``.
    Each loader visits sessions one at a time, keeping RAM usage constant.

    Args:
        cache_dir: Path to cached session .npy files.
        metadata: Metadata dict from ``preprocess_and_cache()``.
        config: Data configuration dict (batch_size, compute settings, etc).

    Returns:
        Dict with keys 'train', 'val', 'test', each a SessionCyclingLoader.
    """
    loaders = {}
    for split in ("train", "val", "test"):
        loader = SessionCyclingLoader(
            cache_dir=cache_dir,
            metadata=metadata,
            split=split,
            config=config,
        )
        loaders[split] = loader
        logger.info(
            "  %s: ~%d estimated samples, ~%d batches (lazy, %d sessions)",
            split, loader._total_samples, len(loader),
            metadata["num_sessions"],
        )

    return loaders


class EagerSessionCyclingLoader:
    """
    Eager variant of SessionCyclingLoader that pre-loads all sessions at init.

    Pre-pads and pre-builds MaskedSpikeCountDataset objects for every session
    during __init__, so __iter__ only creates lightweight DataLoaders — no
    disk I/O, no padding, no dataset construction.  This eliminates the
    1-3 seconds of GPU idle time per session transition that the lazy loader
    incurs.

    Trade-off: uses more RAM (~2-3 GB for 39 Steinmetz sessions padded to
    M_max=1240).  Safe on NRP pods with 32Gi RAM allocation.

    Args:
        cache_dir: Path to the cache directory with .npy files and metadata.
        metadata: Metadata dict from ``preprocess_and_cache()``.
        split: Which split to serve: ``"train"``, ``"val"``, or ``"test"``.
        config: Data configuration dict (for batch_size, etc).
        shuffle_sessions: Whether to shuffle session order each epoch
            (default True for train, False otherwise).
    """

    def __init__(
        self,
        cache_dir: Path,
        metadata: Dict[str, Any],
        split: str,
        config: Dict[str, Any],
        shuffle_sessions: Optional[bool] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.metadata = metadata
        self.split = split
        self.m_max = metadata["m_max"]
        self.history_bins = metadata.get("history_bins", config.get("history_bins", 50))

        # DataLoader config (same platform-aware logic as lazy loader)
        import platform
        self.batch_size = config.get("batch_size", 128)
        if platform.system() == "Windows":
            self.num_workers = 0
            self.pin_memory = False
        else:
            compute_cfg = config.get("compute", {})
            self.num_workers = compute_cfg.get("num_workers", 4)
            self.pin_memory = compute_cfg.get("pin_memory", True)

        # Shuffle sessions each epoch for training by default
        if shuffle_sessions is None:
            self.shuffle_sessions = (split == "train")
        else:
            self.shuffle_sessions = shuffle_sessions

        # Shuffle within-session samples for training
        self.shuffle_samples = (split == "train")

        # --- Pre-load all sessions at init ---
        self._datasets: List[MaskedSpikeCountDataset] = []
        self._session_ids: List[str] = []  # Parallel list: dataset → session_id
        self._total_samples = 0

        # Check if history features were cached
        n_feat = metadata.get("n_features_per_channel", 0)
        # Check if covariates were cached
        n_cov = metadata.get("n_covariates", 0)

        logger.info(
            "EagerSessionCyclingLoader: pre-loading %d sessions for '%s' split...",
            metadata["num_sessions"], split,
        )

        for sess_idx in range(metadata["num_sessions"]):
            sess_info = metadata["sessions"][sess_idx]

            # Load cached count matrix (uint8)
            npy_path = self.cache_dir / f"session_{sess_idx:03d}.npy"
            counts_u8 = np.load(npy_path)  # (M_i, T_i), uint8

            # Slice to requested split
            bounds = sess_info["split_boundaries"]
            t_total = sess_info["num_bins"]
            if split == "train":
                start, end = 0, bounds["train_end"]
            elif split == "val":
                start, end = bounds["train_end"], bounds["val_end"]
            else:  # test
                start, end = bounds["val_end"], t_total

            split_len = end - start
            if split_len <= self.history_bins:
                del counts_u8
                continue

            counts_split = counts_u8[:, start:end].astype(np.int32)
            del counts_u8

            # Pad counts to M_max
            m_i = counts_split.shape[0]
            padded = pad_to_channels(counts_split, self.m_max)
            del counts_split

            # Load and pad history features if they exist
            if n_feat > 0:
                feat_path = self.cache_dir / f"session_{sess_idx:03d}_features.npy"
                if feat_path.exists():
                    features = np.load(feat_path)  # (n_feat * M_i, T_i)
                    features_split = features[:, start:end]
                    del features
                    # Pad features: n_feat blocks of M_i -> n_feat blocks of M_max
                    feat_padded = np.zeros(
                        (n_feat * self.m_max, features_split.shape[1]),
                        dtype=np.float32,
                    )
                    for f in range(n_feat):
                        feat_padded[f * self.m_max : f * self.m_max + m_i, :] = \
                            features_split[f * m_i : (f + 1) * m_i, :]
                    del features_split
                    # Concatenate: (M_max + n_feat * M_max, T_split)
                    padded = np.concatenate(
                        [padded.astype(np.float32), feat_padded], axis=0,
                    )
                    del feat_padded

            # Load and slice covariates if they exist
            cov_split = None
            if n_cov > 0:
                cov_path = self.cache_dir / f"session_{sess_idx:03d}_covariates.npy"
                if cov_path.exists():
                    cov_full = np.load(cov_path)  # (n_cov, T_i), float32
                    cov_split = cov_full[:, start:end]
                    del cov_full

            # Build mask
            mask = build_channel_mask(m_i, self.m_max)
            session_masks = mask.reshape(1, -1)  # (1, M_max)
            mask_index = np.zeros(padded.shape[1], dtype=np.int32)

            # Create dataset with output_channels = M_max
            ds = MaskedSpikeCountDataset(
                spike_counts=padded,
                mask_index=mask_index,
                session_masks=session_masks,
                history_bins=self.history_bins,
                output_channels=self.m_max,
                covariates=cov_split,
            )
            del padded, mask_index, cov_split

            if len(ds) > 0:
                self._datasets.append(ds)
                self._session_ids.append(f"session_{sess_idx:03d}")
                self._total_samples += len(ds)

        self._total_batches = max(1, self._total_samples // self.batch_size)

        # Session-specific head support (Phase 1):
        # - current_session_id: set during iteration, read by trainer
        # - session_neuron_counts: maps session_id → N_i for building model
        # - _session_ids: parallel list matching _datasets for reverse mapping
        self.current_session_id: Optional[str] = None
        self.session_neuron_counts: Dict[str, int] = {
            f"session_{s['index']:03d}": s["num_units"]
            for s in metadata["sessions"]
        }

        logger.info(
            "EagerSessionCyclingLoader: pre-loaded %d sessions, "
            "%d total samples, ~%d batches",
            len(self._datasets), self._total_samples, self._total_batches,
        )

    def __len__(self) -> int:
        """Estimated total batches across all sessions."""
        return self._total_batches

    def __iter__(self):
        """
        Iterate over all pre-loaded sessions, yielding (x, y, mask[, cov]) batches.

        Only creates a lightweight DataLoader per session — all datasets
        are already in memory.  No disk I/O, padding, or dataset
        construction overhead.
        """
        # Determine session order
        session_indices = list(range(len(self._datasets)))
        if self.shuffle_sessions:
            import random
            random.shuffle(session_indices)

        for sess_idx in session_indices:
            ds = self._datasets[sess_idx]

            # Create lightweight DataLoader (no dataset init overhead)
            loader = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=self.shuffle_samples,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=(self.split == "train"),
            )

            for batch in loader:
                # Expose current session identity for session-specific heads
                self.current_session_id = self._session_ids[sess_idx]
                yield batch

            del loader


def create_eager_dataloaders(
    cache_dir: Path,
    metadata: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, "EagerSessionCyclingLoader"]:
    """
    Create train/val/test EagerSessionCyclingLoaders for multi-session training.

    Pre-loads all sessions at creation time, trading RAM for GPU utilization.
    All sessions are padded, masked, and converted to datasets upfront so
    that the training loop has zero I/O overhead between sessions.

    Args:
        cache_dir: Path to cached session .npy files.
        metadata: Metadata dict from ``preprocess_and_cache()``.
        config: Data configuration dict (batch_size, compute settings, etc).

    Returns:
        Dict with keys 'train', 'val', 'test', each an EagerSessionCyclingLoader.
    """
    loaders = {}
    for split in ("train", "val", "test"):
        loader = EagerSessionCyclingLoader(
            cache_dir=cache_dir,
            metadata=metadata,
            split=split,
            config=config,
        )
        loaders[split] = loader
        logger.info(
            "  %s: %d samples, ~%d batches (eager, %d sessions)",
            split, loader._total_samples, len(loader),
            len(loader._datasets),
        )

    return loaders


def create_dataloaders(
    cache_dir: Path,
    metadata: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Unified factory: selects eager or lazy dataloaders based on config.

    Set ``loader_mode: "eager"`` in the data config to pre-load all sessions
    into RAM at init.  Default is ``"lazy"`` (existing behaviour).

    Args:
        cache_dir: Path to cached session .npy files.
        metadata: Metadata dict from ``preprocess_and_cache()``.
        config: Data configuration dict.

    Returns:
        Dict with keys 'train', 'val', 'test', each a cycling loader.
    """
    mode = config.get("loader_mode", "lazy")
    logger.info("Creating dataloaders in '%s' mode", mode)

    if mode == "eager":
        return create_eager_dataloaders(cache_dir, metadata, config)
    elif mode == "lazy":
        return create_lazy_dataloaders(cache_dir, metadata, config)
    else:
        raise ValueError(
            f"Unknown loader_mode '{mode}'. Must be 'eager' or 'lazy'."
        )
