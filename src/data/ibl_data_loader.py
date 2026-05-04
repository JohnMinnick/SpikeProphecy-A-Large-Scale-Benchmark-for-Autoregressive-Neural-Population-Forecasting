"""
IBL Brain-wide Map data loader for spike-count forecasting.

Downloads and loads Neuropixels spike data from the International Brain
Laboratory's Brain-wide Map dataset via the ONE API. Produces MockSorting
objects compatible with the existing pipeline (bin_spike_trains, etc.).

The IBL dataset contains 547+ Neuropixels recordings from mice performing
a standardized decision-making task. The "Repeated Site" subset (91
recordings, 12 labs) uses standardized probe insertion locations, making
it ideal for cross-lab reproducibility claims.

Data access:
    - ONE API: pip install ONE-api
    - Public server: https://openalyx.internationalbrainlab.org
    - AWS S3: s3://ibl-brain-wide-map-public

Usage:
    from src.data.ibl_data_loader import load_ibl_session, list_ibl_sessions

    # List available repeated-site sessions
    eids = list_ibl_sessions(tag="repeated_site")

    # Load one session
    sorting, metadata = load_ibl_session(eid, config)

See ADR-0016 for rationale on adding the IBL dataset.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data.modulated_generator import MockSorting

logger = logging.getLogger(__name__)

# Default IBL server URL (public, no credentials needed)
IBL_BASE_URL = "https://openalyx.internationalbrainlab.org"


def _get_one_client(
    base_url: str = IBL_BASE_URL,
    cache_dir: Optional[str] = None,
):
    """
    Create and return an IBL ONE API client.

    Uses the public Alyx server (no credentials required for read-only
    access to released datasets).

    Args:
        base_url: Alyx server URL. Default: public IBL server.
        cache_dir: Local cache directory for downloaded data.
            Default: data/raw/ibl/.

    Returns:
        ONE client instance.

    Raises:
        ImportError: If ONE-api is not installed.
    """
    try:
        from one.api import ONE
    except ImportError as e:
        raise ImportError(
            "ONE-api is required for IBL data loading. "
            "Install it with: pip install ONE-api\n"
        ) from e

    if cache_dir is None:
        cache_dir = str(Path("data/raw/ibl"))

    # Create cache directory
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Connect to public IBL server (no password needed)
    one = ONE(
        base_url=base_url,
        cache_dir=cache_dir,
        silent=True,
        password="international",  # Public access password
    )
    return one


def list_ibl_sessions(
    tag: str = "repeated_site",
    base_url: str = IBL_BASE_URL,
    cache_dir: Optional[str] = None,
) -> List[str]:
    """
    List available IBL session EIDs for a given data release tag.

    Args:
        tag: Data release tag to filter by. Common options:
            - "repeated_site": 91 recordings, 12 labs (recommended)
            - "2022_Q2_IBL_et_al_RepeatedSite": Same, formal tag name
        base_url: Alyx server URL.
        cache_dir: Local cache directory.

    Returns:
        List of session EID strings.
    """
    one = _get_one_client(base_url, cache_dir)

    # Map friendly tag names to formal IBL tag names
    tag_map = {
        "repeated_site": "2022_Q2_IBL_et_al_RepeatedSite",
        "brain_wide_map": "Brainwidemap",  # 2025 release: 459 sessions
        "brainwidemap": "Brainwidemap",    # Direct alias
    }
    formal_tag = tag_map.get(tag, tag)

    # Use the Alyx REST endpoint — this reliably returns sessions
    # matching both the tag and having spike sorting data available.
    # The one.search() method has version-specific parameter issues.
    try:
        sessions = one.alyx.rest(
            "sessions", "list",
            dataset_types="spikes.times",
            tag=formal_tag,
        )
        eids = [s["id"] for s in sessions]
    except Exception as e:
        logger.warning(
            "REST search failed (%s), falling back to one.search().", e,
        )
        # Fallback: load cache by tag and search locally
        try:
            one.load_cache(tag=formal_tag)
        except Exception:
            pass
        eids = list(one.search(datasets="spikes.times"))

    logger.info(
        "Found %d IBL sessions for tag '%s'", len(eids), tag,
    )
    return eids


def load_ibl_session(
    eid: str,
    config: Dict[str, Any],
    base_url: str = IBL_BASE_URL,
    cache_dir: Optional[str] = None,
) -> Tuple[MockSorting, Dict[str, Any]]:
    """
    Load spike data from a single IBL session via the ONE API.

    Downloads spikes.times and spikes.clusters, groups by cluster,
    filters by quality and firing rate, and wraps in a MockSorting
    compatible with our pipeline.

    Args:
        eid: IBL session Experiment ID (UUID string).
        config: Configuration dictionary with keys:
            - seed (int): Random seed for reproducibility.
            - nwb.sampling_frequency (float): Target sampling freq (Hz).
            - nwb.min_firing_rate_hz (float): Min firing rate threshold.
            - nwb.max_units (int|null): Max units to retain.
            - nwb.quality_labels (list|null): Quality labels to accept.
            - nwb.duration_limit_s (float|null): Duration truncation limit.
        base_url: Alyx server URL.
        cache_dir: Local cache directory.

    Returns:
        Tuple of:
            - MockSorting: Spike train object compatible with bin_spike_trains.
            - metadata: Dict with session info, unit details, and filter stats.

    Raises:
        ImportError: If ONE-api is not installed.
        ValueError: If no units remain after filtering.
    """
    one = _get_one_client(base_url, cache_dir)

    seed = config.get("seed", 42)
    rng = np.random.default_rng(seed)

    nwb_config = config.get("nwb", {})
    sampling_frequency = nwb_config.get("sampling_frequency", 30000.0)
    min_firing_rate_hz = nwb_config.get("min_firing_rate_hz", 1.0)
    max_units = nwb_config.get("max_units", None)
    quality_labels = nwb_config.get("quality_labels", ["good"])
    duration_limit_s = nwb_config.get("duration_limit_s", None)

    logger.info("Loading IBL session: %s", eid)

    # --- Discover available probes ---
    # IBL stores spike sorting results in probe-specific pykilosort
    # collections: alf/probe00/pykilosort, alf/probe01/pykilosort, etc.
    collections = one.list_collections(eid)
    probe_collections = sorted([
        c for c in collections
        if c.startswith("alf/probe") and c.endswith("/pykilosort")
    ])

    # Fallback: try direct alf/probeXX collections if no pykilosort
    if not probe_collections:
        probe_collections = sorted([
            c for c in collections
            if c.startswith("alf/probe") and "/" not in c[len("alf/"):]
        ])

    if not probe_collections:
        raise ValueError(
            f"No probe collections found for session {eid}. "
            f"Available collections: {collections}"
        )

    logger.info("  Found %d probes: %s", len(probe_collections), probe_collections)

    # --- Download spike data from each probe ---
    all_spike_times = []
    all_spike_clusters = []
    cluster_offset = 0  # Offset cluster IDs across probes to avoid collisions

    all_quality = []
    all_regions = []

    for probe_col in probe_collections:
        logger.info("  Loading probe: %s", probe_col)

        # Use load_object to get all spike-related arrays at once
        # This is the recommended IBL API for spike sorting data
        spikes = one.load_object(eid, "spikes", collection=probe_col)
        probe_st = spikes["times"]
        probe_cl = spikes["clusters"]

        # Offset cluster IDs to avoid collisions across probes
        n_clusters_this_probe = int(probe_cl.max()) + 1 if len(probe_cl) > 0 else 0
        probe_cl_offset = probe_cl + cluster_offset

        all_spike_times.append(probe_st)
        all_spike_clusters.append(probe_cl_offset)

        # Try to load cluster quality and region labels
        try:
            clusters = one.load_object(eid, "clusters", collection=probe_col)
            # Cluster metrics: look for 'label' column
            if "metrics" in clusters and hasattr(clusters["metrics"], "get"):
                labels = list(clusters["metrics"].get("label", []))
                all_quality.extend(labels)
            else:
                all_quality.extend(["unknown"] * n_clusters_this_probe)
            # Brain region acronyms
            if "acronym" in clusters:
                all_regions.extend(list(clusters["acronym"]))
            else:
                all_regions.extend(["unknown"] * n_clusters_this_probe)
        except Exception:
            all_quality.extend(["unknown"] * n_clusters_this_probe)
            all_regions.extend(["unknown"] * n_clusters_this_probe)

        logger.info(
            "    %s: %d spikes, %d clusters (offset +%d)",
            probe_col, len(probe_st), n_clusters_this_probe, cluster_offset,
        )

        cluster_offset += n_clusters_this_probe

    # Concatenate spike data across probes
    spike_times = np.concatenate(all_spike_times)
    spike_clusters = np.concatenate(all_spike_clusters).astype(np.int64)

    # Set quality/region data if any non-unknown labels exist
    cluster_quality = (
        all_quality if any(q != "unknown" for q in all_quality) else None
    )
    cluster_regions = (
        all_regions if any(r != "unknown" for r in all_regions) else None
    )

    # --- Group spikes by cluster ---
    unique_clusters = np.unique(spike_clusters)
    num_raw_units = len(unique_clusters)
    logger.info("  Session has %d clusters", num_raw_units)

    # Build per-cluster spike time arrays
    spike_times_list = []
    unit_indices = []
    quality_list = []
    region_list = []

    for cluster_id in unique_clusters:
        # Extract spike times for this cluster
        cluster_mask = spike_clusters == cluster_id
        st = spike_times[cluster_mask]

        # Sort spike times (should already be sorted, but be safe)
        st = np.sort(st).astype(np.float64)

        spike_times_list.append(st)
        unit_indices.append(int(cluster_id))

        # Map quality and region labels (if available)
        if cluster_quality is not None and cluster_id < len(cluster_quality):
            quality_list.append(str(cluster_quality[int(cluster_id)]))
        else:
            quality_list.append("unknown")

        if cluster_regions is not None and cluster_id < len(cluster_regions):
            region_list.append(str(cluster_regions[int(cluster_id)]))
        else:
            region_list.append("unknown")

    # --- Determine recording duration ---
    all_max_times = [st[-1] for st in spike_times_list if len(st) > 0]
    recording_duration_s = max(all_max_times) if all_max_times else 0.0

    # Apply duration limit
    if duration_limit_s is not None and duration_limit_s < recording_duration_s:
        logger.info(
            "  Truncating from %.1fs to %.1fs",
            recording_duration_s, duration_limit_s,
        )
        recording_duration_s = duration_limit_s
        spike_times_list = [
            st[st <= duration_limit_s] for st in spike_times_list
        ]

    # --- Filter units ---
    from src.data.real_data_loader import filter_units

    spike_times_list, unit_indices, filter_stats = filter_units(
        spike_times_list=spike_times_list,
        unit_indices=unit_indices,
        sampling_frequency=sampling_frequency,
        duration_s=recording_duration_s,
        min_firing_rate_hz=min_firing_rate_hz,
        max_units=max_units,
        quality_labels=(
            quality_labels if cluster_quality is not None else None
        ),
        quality_list=quality_list if cluster_quality is not None else None,
        brain_region=nwb_config.get("brain_region", None),
        brain_region_list=region_list if cluster_regions is not None else None,
        rng=rng,
    )

    if len(spike_times_list) == 0:
        raise ValueError(
            f"No units remain after filtering for session {eid}. "
            "Try relaxing quality_labels, min_firing_rate_hz, or brain_region."
        )

    # --- Convert to sample indices and build MockSorting ---
    from src.data.real_data_loader import _spike_times_to_samples

    max_sample = int(recording_duration_s * sampling_frequency)
    spike_trains = {}
    unit_rates_info = []

    for new_id, (st, orig_idx) in enumerate(
        zip(spike_times_list, unit_indices)
    ):
        samples = _spike_times_to_samples(st, sampling_frequency, max_sample)
        spike_trains[new_id] = samples

        actual_rate = (
            len(samples) / recording_duration_s
            if recording_duration_s > 0 else 0.0
        )
        unit_rates_info.append({
            "unit_id": new_id,
            "original_cluster_id": orig_idx,
            "actual_rate_hz": float(actual_rate),
            "num_spikes": len(samples),
        })

    sorting = MockSorting(spike_trains, sampling_frequency)

    # --- Summary ---
    total_spikes = sum(info["num_spikes"] for info in unit_rates_info)
    mean_rate = np.mean([info["actual_rate_hz"] for info in unit_rates_info])
    num_units = len(spike_trains)

    logger.info(
        "  Loaded %d units, %d spikes, mean rate=%.2f Hz, duration=%.1fs",
        num_units, total_spikes, mean_rate, recording_duration_s,
    )

    # --- Build metadata ---
    metadata = {
        "source": "ibl",
        "eid": eid,
        "dataset": "ibl_brain_wide_map",
        "seed": seed,
        "num_units": num_units,
        "num_raw_units": num_raw_units,
        "duration_s": recording_duration_s,
        "sampling_frequency": sampling_frequency,
        "total_spikes": total_spikes,
        "mean_rate_hz": float(mean_rate),
        "filter_stats": filter_stats,
        "unit_details": unit_rates_info,
        "brain_regions": (
            filter_stats.get("brain_region_list", None)
        ),
    }

    return sorting, metadata


def load_ibl_multi_session(
    eids: List[str],
    config: Dict[str, Any],
    base_url: str = IBL_BASE_URL,
    cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load and preprocess multiple IBL sessions into cached .npy files.

    This is the IBL equivalent of load_multi_session_nwb — it downloads
    each session, bins spikes, and saves to the session cache. The cached
    files are then compatible with the lazy cycling loader.

    Args:
        eids: List of IBL session EIDs to process.
        config: Data configuration dictionary (same format as Steinmetz).
        base_url: Alyx server URL.
        cache_dir: Local cache for ONE API downloads.

    Returns:
        Tuple of:
            - spike_counts: (M_max, T_total), concatenated count matrix
            - mask_index: (T_total,), session index per bin
            - metadata: dict with session info, m_max, masks
    """
    from src.data.binning import bin_spike_trains
    from src.data.multi_session_loader import (
        pad_to_channels, build_channel_mask,
    )

    bin_width_ms = config.get("bin_width_ms", 50.0)
    history_bins = config.get("history_bins", 10)

    # --- Process each session ---
    session_counts = []
    session_real_m = []
    session_details = []

    for i, eid in enumerate(eids):
        logger.info(
            "[%d/%d] Loading IBL session: %s", i + 1, len(eids), eid,
        )

        sorting, load_meta = load_ibl_session(
            eid, config, base_url, cache_dir,
        )

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
            "eid": eid,
            "num_units": m_i,
            "num_bins": t_i,
            "duration_s": t_i * bin_width_ms / 1000,
            "load_metadata": load_meta,
            "bin_metadata": bin_meta,
        })

    # --- Determine M_max ---
    m_max = max(session_real_m)
    logger.info(
        "IBL M_max = %d (range: %d to %d across %d sessions)",
        m_max, min(session_real_m), m_max, len(session_counts),
    )

    # --- Build masks, pad, concatenate ---
    num_sessions = len(session_counts)
    session_masks = np.zeros((num_sessions, m_max), dtype=np.float32)
    for i, real_m in enumerate(session_real_m):
        session_masks[i] = build_channel_mask(real_m, m_max)

    padded_segments = []
    mask_segments = []

    for i, counts in enumerate(session_counts):
        padded = pad_to_channels(counts, m_max)
        t_i = padded.shape[1]
        padded_segments.append(padded)
        mask_segments.append(np.full(t_i, i, dtype=np.int32))

        if i < num_sessions - 1:
            gap = np.zeros((m_max, history_bins), dtype=padded.dtype)
            padded_segments.append(gap)
            mask_segments.append(
                np.full(history_bins, -1, dtype=np.int32)
            )

    spike_counts = np.concatenate(padded_segments, axis=1)
    mask_index = np.concatenate(mask_segments)

    t_total = spike_counts.shape[1]
    total_gap_bins = (
        history_bins * (num_sessions - 1) if num_sessions > 1 else 0
    )

    logger.info(
        "IBL concatenated: shape (%d, %d) — %d data + %d gap bins",
        m_max, t_total, t_total - total_gap_bins, total_gap_bins,
    )

    metadata = {
        "source": "ibl_multi",
        "dataset": "ibl_brain_wide_map",
        "num_sessions": num_sessions,
        "m_max": m_max,
        "session_real_m": session_real_m,
        "session_masks": session_masks,
        "total_bins": t_total,
        "total_gap_bins": total_gap_bins,
        "gap_bins": history_bins,
        "session_details": session_details,
    }

    return spike_counts, mask_index, metadata


def preprocess_and_cache_ibl(
    config: Dict[str, Any],
    cache_dir: Optional[str] = None,
    force_reprocess: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Preprocess IBL sessions: download via ONE API, bin, and cache as .npy.

    This is the IBL equivalent of multi_session_loader.preprocess_and_cache.
    Produces the same output format (session_NNN.npy + metadata.json) so
    the lazy cycling DataLoader works without modification.

    Args:
        config: Data configuration dictionary with keys:
            - ibl.tag: IBL data release tag (e.g., "repeated_site")
            - ibl.max_sessions: Maximum sessions to process
            - ibl.cache_dir: Local ONE API cache directory
            - bin_width_ms: Bin width in milliseconds
            - history_bins: Number of history bins
            - splits: Train/val/test split ratios
        cache_dir: Override for output cache directory.
            Default: "data/processed/ibl_session_cache/"
        force_reprocess: If True, reprocess even if cache exists.

    Returns:
        Tuple of:
            - cache_path: Path to the cache directory
            - metadata: dict with m_max, session info, split boundaries
    """
    import json
    from src.data.binning import bin_spike_trains

    ibl_cfg = config.get("ibl", {})
    tag = ibl_cfg.get("tag", "repeated_site")
    max_sessions = ibl_cfg.get("max_sessions", 3)
    one_cache_dir = ibl_cfg.get("cache_dir", "data/raw/ibl")

    bin_width_ms = config.get("bin_width_ms", 50.0)
    history_bins = config.get("history_bins", 10)
    splits = config.get("splits", {"train": 0.7, "val": 0.15, "test": 0.15})

    # --- Determine output cache directory ---
    if cache_dir:
        cache_path = Path(cache_dir)
    else:
        cache_path = Path("data/processed/ibl_session_cache")

    metadata_file = cache_path / "metadata.json"

    # --- Check if cache already exists ---
    if metadata_file.exists() and not force_reprocess:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        logger.info(
            "IBL cache found at %s with %d sessions (M_max=%d). "
            "Skipping preprocessing.",
            cache_path, metadata["num_sessions"], metadata["m_max"],
        )
        return cache_path, metadata

    # --- Discover available sessions ---
    eids = list_ibl_sessions(
        tag=tag, cache_dir=one_cache_dir,
    )
    selected_eids = eids[:max_sessions]
    logger.info(
        "IBL: preprocessing %d sessions (of %d available, tag=%s) → %s",
        len(selected_eids), len(eids), tag, cache_path,
    )

    # Create cache directory
    cache_path.mkdir(parents=True, exist_ok=True)

    # --- Process each session ---
    session_info: List[Dict[str, Any]] = []
    overflow_total = 0

    for i, eid in enumerate(selected_eids):
        logger.info(
            "[%d/%d] Processing IBL session: %s",
            i + 1, len(selected_eids), eid,
        )

        # Load spike trains via ONE API
        try:
            sorting, load_meta = load_ibl_session(
                eid, config, cache_dir=one_cache_dir,
            )
        except Exception as e:
            logger.error("  [!] Failed to load session %s. Skipping. Error: %s", eid, e)
            continue

        # Bin spike trains
        counts, bin_meta = bin_spike_trains(
            sorting, bin_width_ms=bin_width_ms,
        )
        m_i, t_i = counts.shape

        # Check for uint8 overflow (any count > 255)
        overflow_count = int(np.sum(counts > 255))
        if overflow_count > 0:
            overflow_total += overflow_count
            logger.warning(
                "  Session %d: %d bins exceed uint8 range (max=%d). "
                "Clamping to 255.",
                i, overflow_count, int(counts.max()),
            )

        # Save as uint8 (same format as NWB cache)
        counts_u8 = np.clip(counts, 0, 255).astype(np.uint8)
        npy_path = cache_path / f"session_{i:03d}.npy"
        np.save(npy_path, counts_u8)

        # Compute per-session split boundaries
        train_end = int(t_i * splits["train"])
        val_end = train_end + int(t_i * splits["val"])

        # Extract brain regions from load metadata
        brain_regions = load_meta.get("brain_regions", None)

        session_info.append({
            "index": i,
            "file": eid,  # Use EID as the file identifier
            "npy_file": str(npy_path),
            "num_units": m_i,
            "num_bins": t_i,
            "duration_s": round(t_i * bin_width_ms / 1000, 2),
            "split_boundaries": {
                "train_end": train_end,
                "val_end": val_end,
            },
            "brain_regions": brain_regions,
        })

        logger.info(
            "  Session %d: %d units, %d bins (%.1fs) → %s",
            i, m_i, t_i, t_i * bin_width_ms / 1000, npy_path.name,
        )

    # --- Compute M_max across all sessions ---
    all_m = [s["num_units"] for s in session_info]
    m_max = max(all_m)

    logger.info(
        "IBL M_max = %d (range: %d to %d across %d sessions)",
        m_max, min(all_m), m_max, len(session_info),
    )

    if overflow_total > 0:
        logger.warning(
            "Total uint8 overflows across all IBL sessions: %d",
            overflow_total,
        )

    # --- Write metadata (same format as NWB preprocess_and_cache) ---
    metadata = {
        "num_sessions": len(session_info),
        "m_max": m_max,
        "history_bins": history_bins,
        "bin_width_ms": bin_width_ms,
        "n_features_per_channel": 0,  # IBL baseline: no history features
        "n_covariates": 0,            # IBL baseline: no covariates
        "covariate_names": [],
        "sessions": session_info,
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "IBL cache complete: %d sessions, M_max=%d → %s",
        len(session_info), m_max, cache_path,
    )

    return cache_path, metadata
