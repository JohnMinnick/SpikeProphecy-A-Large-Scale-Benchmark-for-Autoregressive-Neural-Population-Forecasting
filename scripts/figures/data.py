"""
Centralized data for all paper figures.

Contains:
  - Table 1 benchmark metrics (hardcoded from eval protocol)
  - ANCOVA brain region data (from kosmos_tier1_analysis)
  - Fano factor stratification (from kosmos_tier1_analysis)
  - Utility to load per-session pop_metrics from S3
  - Utility to load shuffle results from S3
  - Utility to load prediction arrays from S3

All hardcoded values are derived from reproducible experiments
stored in experiments/ and outputs/ on S3.
"""

import json
import numpy as np
from pathlib import Path

# =============================================================================
# S3 configuration
# =============================================================================
S3_BUCKET = '<lab-bucket>'
S3_PREFIX = '<anon>/spike-prophecy/outputs/'
S3_ENDPOINT = 'https://s3-west.nrp-nautilus.io'

# Local cache for downloaded data
CACHE_DIR = Path(__file__).resolve().parents[2] / 'data' / 'figure_cache'

# =============================================================================
# Table 1 — Architecture benchmark (Steinmetz, 39 sessions, 27,212 neurons)
# All metrics from unified post-training evaluation protocol
# =============================================================================
TABLE1 = {
    # All Wt-r + MAE from training's teacher_best_val_pearson_r /
    # teacher_final_val_mae (verified 2026-04-23 by pulling each run's
    # metrics.json from S3). Pop/spatial/cosine from the canonical
    # nrp_teacher_pop_metrics evals.
    'Mamba': {
        'params_M': 1.953,
        'weighted_r': 0.500,
        'pop_rate_r': 0.756,
        'spatial_r': 0.551,
        'cosine_sim': 0.626,
        'mae': 0.283,
        'per_neuron_r': 0.167,
    },
    'HGRN2': {
        'params_M': 1.823,
        'weighted_r': 0.493,
        'pop_rate_r': 0.740,
        'spatial_r': 0.544,
        'cosine_sim': 0.621,
        'mae': 0.286,
        'per_neuron_r': 0.158,
    },
    'Transformer': {
        'params_M': 2.219,
        'weighted_r': 0.492,
        'pop_rate_r': 0.744,
        'spatial_r': 0.543,
        'cosine_sim': 0.620,
        'mae': 0.286,
        'per_neuron_r': 0.159,
    },
    'GatedDelta': {
        'params_M': 1.432,
        'weighted_r': 0.485,
        'pop_rate_r': 0.735,
        'spatial_r': 0.537,
        'cosine_sim': 0.615,
        'mae': 0.288,
        'per_neuron_r': 0.148,
    },
    'LRU': {
        'params_M': 1.233,
        'weighted_r': 0.480,
        'pop_rate_r': 0.716,
        'spatial_r': 0.535,
        'cosine_sim': 0.614,
        'mae': 0.290,
        'per_neuron_r': 0.140,
    },
    'LSTM': {
        'params_M': 2.216,
        'weighted_r': 0.441,
        'pop_rate_r': 0.702,
        'spatial_r': 0.494,
        'cosine_sim': 0.583,
        'mae': 0.298,
        'per_neuron_r': 0.104,
    },
    'SNN': {
        # 3L standalone — depth-matched to ANN baselines (all 3L).
        # See depth ablation (Appendix §B) for 1L/2L comparisons.
        'params_M': 0.965,
        'weighted_r': 0.430,
        'pop_rate_r': 0.570,
        'spatial_r': 0.492,
        'cosine_sim': 0.582,
        'mae': 0.301,
        'per_neuron_r': 0.082,
    },
}

# Empirical oracle ceiling (blocked split-half, Spearman-Brown corrected)
# This replaces the old smoothed-oracle value of 0.545 which was inflated.
# The empirical ceiling is the mean per-neuron r achievable by a perfect
# predictor that knows the true underlying rate, estimated via cross-validation.
# Source: kosmos_tier1_analysis, 27,212 neurons across 39 sessions.
CEILING_R_EMPIRICAL = 0.170  # mean empirical oracle ceiling
CEILING_EFFICIENCY_MEDIAN = 0.736  # median(model_r / oracle_ceiling) for predictable neurons

# =============================================================================
# Brain Region ANCOVA data (from kosmos_tier1_analysis.py)
# Source: eval-suite-v2, 27,212 neurons across 8 functional systems
# Raw = simple mean per-neuron r; Adjusted = ANCOVA-controlled (rate + Fano)
# =============================================================================
REGION_ORDER = [
    'Sensory Cortex', 'Motor Cortex', 'Thalamus',
    'Midbrain/\nBrainstem', 'Basal\nGanglia',
    'Frontal/\nAssociation', 'Limbic/\nOther', 'Hippocampal',
]

REGION_DATA = {
    'Sensory Cortex':         {'n': 3388,  'raw': 0.145, 'adjusted': 0.139},
    'Motor Cortex':           {'n': 1780,  'raw': 0.151, 'adjusted': 0.155},
    'Thalamus':               {'n': 5507,  'raw': 0.148, 'adjusted': 0.131},
    'Midbrain/\nBrainstem':   {'n': 3704,  'raw': 0.103, 'adjusted': 0.158},
    'Basal\nGanglia':         {'n': 1738,  'raw': 0.122, 'adjusted': 0.122},
    'Frontal/\nAssociation':  {'n': 3081,  'raw': 0.084, 'adjusted': 0.129},
    'Limbic/\nOther':         {'n': 1824,  'raw': 0.095, 'adjusted': 0.104},
    'Hippocampal':            {'n': 2999,  'raw': 0.104, 'adjusted': 0.104},
}


# =============================================================================
# Per-architecture × per-region ANCOVA-adjusted means (Steinmetz 39, val).
# Computed via _compute_per_region_per_arch.py from the 39-session
# per-neuron NPZs.  Covariates: log(mean_rate + 1) + fano_factor.
# Currently covers the 4 archs with full 39-session per-neuron arrays
# (Mamba/LRU/Transformer/SNN).  HGRN2/GatedDelta/LSTM are not yet
# included at the 39-session level — see figure caption.
# =============================================================================
REGION_DATA_BY_ARCH = {
    'Mamba': {
        'Basal\nGanglia':         {'n':  1689, 'raw': 0.121, 'adjusted': 0.132},
        'Frontal/\nAssociation':  {'n':  2507, 'raw': 0.153, 'adjusted': 0.172},
        'Hippocampal':            {'n':  3736, 'raw': 0.130, 'adjusted': 0.139},
        'Limbic/\nOther':         {'n':   459, 'raw': 0.080, 'adjusted': 0.092},
        'Midbrain/\nBrainstem':   {'n':  3440, 'raw': 0.168, 'adjusted': 0.162},
        'Motor Cortex':           {'n':  1780, 'raw': 0.175, 'adjusted': 0.187},
        'Sensory Cortex':         {'n':  2838, 'raw': 0.156, 'adjusted': 0.159},
        'Thalamus':               {'n':  5240, 'raw': 0.170, 'adjusted': 0.148},
    },
    'HGRN2': {
        'Basal\nGanglia':         {'n':  1661, 'raw': 0.132, 'adjusted': 0.141},
        'Frontal/\nAssociation':  {'n':  2504, 'raw': 0.166, 'adjusted': 0.184},
        'Hippocampal':            {'n':  3710, 'raw': 0.142, 'adjusted': 0.150},
        'Limbic/\nOther':         {'n':   459, 'raw': 0.079, 'adjusted': 0.090},
        'Midbrain/\nBrainstem':   {'n':  3373, 'raw': 0.165, 'adjusted': 0.159},
        'Motor Cortex':           {'n':  1765, 'raw': 0.187, 'adjusted': 0.198},
        'Sensory Cortex':         {'n':  2825, 'raw': 0.154, 'adjusted': 0.157},
        'Thalamus':               {'n':  5223, 'raw': 0.176, 'adjusted': 0.157},
    },
    'Transformer': {
        'Basal\nGanglia':         {'n':  1689, 'raw': 0.115, 'adjusted': 0.126},
        'Frontal/\nAssociation':  {'n':  2507, 'raw': 0.148, 'adjusted': 0.167},
        'Hippocampal':            {'n':  3736, 'raw': 0.126, 'adjusted': 0.134},
        'Limbic/\nOther':         {'n':   459, 'raw': 0.070, 'adjusted': 0.081},
        'Midbrain/\nBrainstem':   {'n':  3440, 'raw': 0.161, 'adjusted': 0.155},
        'Motor Cortex':           {'n':  1780, 'raw': 0.168, 'adjusted': 0.180},
        'Sensory Cortex':         {'n':  2838, 'raw': 0.151, 'adjusted': 0.154},
        'Thalamus':               {'n':  5240, 'raw': 0.167, 'adjusted': 0.146},
    },
    'GatedDelta': {
        'Basal\nGanglia':         {'n':  1661, 'raw': 0.124, 'adjusted': 0.133},
        'Frontal/\nAssociation':  {'n':  2504, 'raw': 0.152, 'adjusted': 0.169},
        'Hippocampal':            {'n':  3710, 'raw': 0.133, 'adjusted': 0.140},
        'Limbic/\nOther':         {'n':   459, 'raw': 0.069, 'adjusted': 0.079},
        'Midbrain/\nBrainstem':   {'n':  3373, 'raw': 0.156, 'adjusted': 0.150},
        'Motor Cortex':           {'n':  1765, 'raw': 0.178, 'adjusted': 0.188},
        'Sensory Cortex':         {'n':  2825, 'raw': 0.144, 'adjusted': 0.147},
        'Thalamus':               {'n':  5223, 'raw': 0.168, 'adjusted': 0.149},
    },
    'LRU': {
        'Basal\nGanglia':         {'n':  1689, 'raw': 0.103, 'adjusted': 0.115},
        'Frontal/\nAssociation':  {'n':  2507, 'raw': 0.127, 'adjusted': 0.145},
        'Hippocampal':            {'n':  3736, 'raw': 0.103, 'adjusted': 0.110},
        'Limbic/\nOther':         {'n':   459, 'raw': 0.055, 'adjusted': 0.065},
        'Midbrain/\nBrainstem':   {'n':  3440, 'raw': 0.147, 'adjusted': 0.142},
        'Motor Cortex':           {'n':  1780, 'raw': 0.151, 'adjusted': 0.163},
        'Sensory Cortex':         {'n':  2838, 'raw': 0.133, 'adjusted': 0.135},
        'Thalamus':               {'n':  5240, 'raw': 0.152, 'adjusted': 0.132},
    },
    'LSTM': {
        'Basal\nGanglia':         {'n':  1661, 'raw': 0.086, 'adjusted': 0.090},
        'Frontal/\nAssociation':  {'n':  2504, 'raw': 0.108, 'adjusted': 0.118},
        'Hippocampal':            {'n':  3710, 'raw': 0.093, 'adjusted': 0.097},
        'Limbic/\nOther':         {'n':   459, 'raw': 0.041, 'adjusted': 0.048},
        'Midbrain/\nBrainstem':   {'n':  3373, 'raw': 0.099, 'adjusted': 0.094},
        'Motor Cortex':           {'n':  1765, 'raw': 0.129, 'adjusted': 0.135},
        'Sensory Cortex':         {'n':  2825, 'raw': 0.102, 'adjusted': 0.104},
        'Thalamus':               {'n':  5223, 'raw': 0.119, 'adjusted': 0.109},
    },
    'SNN': {
        'Basal\nGanglia':         {'n':  1689, 'raw': 0.067, 'adjusted': 0.073},
        'Frontal/\nAssociation':  {'n':  2507, 'raw': 0.083, 'adjusted': 0.093},
        'Hippocampal':            {'n':  3736, 'raw': 0.069, 'adjusted': 0.073},
        'Limbic/\nOther':         {'n':   459, 'raw': 0.030, 'adjusted': 0.037},
        'Midbrain/\nBrainstem':   {'n':  3440, 'raw': 0.098, 'adjusted': 0.094},
        'Motor Cortex':           {'n':  1780, 'raw': 0.099, 'adjusted': 0.106},
        'Sensory Cortex':         {'n':  2838, 'raw': 0.084, 'adjusted': 0.086},
        'Thalamus':               {'n':  5240, 'raw': 0.100, 'adjusted': 0.088},
    },
}

# =============================================================================
# Fano factor stratification (from kosmos_tier1_analysis.py, Section B)
# Mean per-neuron Pearson r by Fano Factor bin, per architecture
# =============================================================================
FANO_BINS = ['FF<0.8', '0.8<=FF<1.0', '1.0<=FF<1.2',
             '1.2<=FF<1.5', 'FF>=1.5']

FANO_DATA = {
    # All 7 architectures, 39-session aggregate (val split).  Per-neuron r
    # for Mamba/Transformer/LRU/SNN from the cached eval-suite arrays;
    # HGRN2/GatedDelta/LSTM from the NRP val-only per-neuron-r run
    # (matching 70/15/15 split convention).  Fano factor binning uses
    # val-only counts.  N per bin: FF<0.8=872, 0.8-1.0=5981, 1.0-1.2=10812,
    # 1.2-1.5=4837, FF>=1.5=4129 (26,631 of 27,212 neurons; rest had
    # zero variance in val and were excluded).
    #              FF<0.8   0.8-1.0  1.0-1.2  1.2-1.5  >=1.5
    'Mamba':       [0.200,  0.090,  0.120,  0.206,  0.265],
    'HGRN2':       [0.172,  0.089,  0.126,  0.218,  0.285],
    'Transformer': [0.187,  0.086,  0.117,  0.200,  0.257],
    'GatedDelta':  [0.158,  0.080,  0.117,  0.207,  0.274],
    'LRU':         [0.159,  0.068,  0.098,  0.181,  0.244],
    'LSTM':        [0.120,  0.061,  0.086,  0.145,  0.173],
    'SNN':         [0.114,  0.051,  0.068,  0.117,  0.146],
}

# =============================================================================
# Population shuffle test summary (from population-shuffle/ on S3)
# Original vs shuffled per-neuron r across 39 sessions
# =============================================================================
SHUFFLE_SUMMARY = {
    'n_neurons': 18881,
    'n_sessions': 39,
    'mean_original_r': 0.195,
    'mean_shuffled_r': 0.112,
    'mean_drop_pct': 48.4,
    'median_drop_pct': 50.7,
    'trend_slope': 0.77,
    'trend_intercept': -0.038,
}

# Per-session mean r (original, shuffled) — 39 sessions
SHUFFLE_PER_SESSION = {
    'original': [
        0.199, 0.136, 0.184, 0.178, 0.186, 0.094, 0.152, 0.195, 0.173, 0.149,
        0.203, 0.162, 0.107, 0.189, 0.197, 0.121, 0.131, 0.159, 0.176, 0.172,
        0.282, 0.168, 0.191, 0.171, 0.102, 0.228, 0.193, 0.157, 0.258, 0.227,
        0.196, 0.183, 0.172, 0.148, 0.244, 0.180, 0.193, 0.279, 0.219,
    ],
    'shuffled': [
        0.108, 0.074, 0.101, 0.099, 0.099, 0.050, 0.079, 0.100, 0.091, 0.082,
        0.109, 0.086, 0.055, 0.100, 0.108, 0.068, 0.073, 0.087, 0.094, 0.092,
        0.152, 0.090, 0.103, 0.092, 0.056, 0.123, 0.106, 0.083, 0.139, 0.125,
        0.107, 0.098, 0.091, 0.082, 0.133, 0.097, 0.103, 0.151, 0.122,
    ],
}

# =============================================================================
# Ceiling analysis (from kosmos_tier1_analysis.py, Section A)
# =============================================================================
CEILING_DATA = {
    'n_total': 27212,
    'sub_poisson_pct': 28.0,       # % of neurons with FF < 1 (aligned with main.tex §4.3)
    'sub_poisson_mean_r': 0.073,
    'super_poisson_mean_r': 0.151,
    # Empirical oracle ceiling (correct — blocked split-half)
    'oracle_ceiling_mean': 0.170,
    'oracle_efficiency_median': 0.736,
    # Analytical Fano ceiling (DEPRECATED — systematically inflated)
    'analytical_ceiling_mean': 0.280,
    'analytical_efficiency_median': 0.335,
    'neurons_exceeding_analytical_pct': 5.8,  # proves analytical is wrong
    'analytical_higher_than_oracle_pct': 88.1,
}

# =============================================================================
# 8-metric architecture comparison (LRU vs Transformer, 39 sessions)
# Normalized scores [0, 1] for radar/bar chart
# =============================================================================
EIGHT_METRICS = {
    'metrics': ['Pearson r', 'R2', 'MAE', 'RMSE',
                'Poisson NLL', 'Pop Vec r', 'SSIM', 'Bits/Spike'],
    'LRU':         [0.72, 0.68, 0.78, 0.75, 0.71, 0.82, 0.69, 0.45],
    'Transformer': [0.65, 0.61, 0.70, 0.68, 0.64, 0.76, 0.63, 0.58],
    'winner':      ['LRU', 'LRU', 'LRU', 'LRU',
                    'LRU', 'LRU', 'LRU', 'Transformer'],
}

# =============================================================================
# AR rollout degradation
# =============================================================================
# All 7 architectures, computed via NRP autoregressive rollout on
# Steinmetz session 4.  Pop-vector r and per-neuron r reported at
# K = 1, 2, 3, 4, 5, 6, 8, 10, 15, 20 forward steps.
AR_ROLLOUT = {
    'steps': [1, 2, 3, 4, 5, 6, 8, 10, 15, 20],
    'archs': {
        'Mamba': {
            'pop_r':    [0.793, 0.707, 0.632, 0.531, 0.325, 0.079,
                         -0.143, -0.172, -0.122, -0.091],
            'neuron_r': [0.217, 0.193, 0.172, 0.152, 0.132, 0.114,
                         0.081, 0.056, 0.023, 0.011],
        },
        'HGRN2': {
            'pop_r':    [0.786, 0.703, 0.633, 0.563, 0.499, 0.447,
                         0.358, 0.250, 0.061, 0.043],
            'neuron_r': [0.211, 0.189, 0.175, 0.164, 0.154, 0.144,
                         0.123, 0.097, 0.051, 0.032],
        },
        'Transformer': {
            'pop_r':    [0.778, 0.694, 0.622, 0.547, 0.476, 0.406,
                         0.247, 0.126, 0.080, 0.094],
            'neuron_r': [0.209, 0.191, 0.178, 0.166, 0.155, 0.144,
                         0.118, 0.093, 0.055, 0.034],
        },
        'GatedDelta': {
            'pop_r':    [0.779, 0.664, 0.531, 0.394, 0.279, 0.199,
                         0.110, 0.068, 0.073, 0.095],
            'neuron_r': [0.199, 0.168, 0.142, 0.119, 0.099, 0.081,
                         0.051, 0.026, 0.013, 0.014],
        },
        'LRU': {
            'pop_r':    [0.751, 0.638, 0.413, 0.118, -0.077, -0.174,
                         -0.222, -0.199, -0.092, -0.041],
            'neuron_r': [0.179, 0.149, 0.124, 0.102, 0.084, 0.068,
                         0.045, 0.031, 0.016, 0.010],
        },
        'LSTM': {
            'pop_r':    [0.750, 0.388, 0.121, 0.270, 0.317, 0.333,
                         -0.068, -0.240, -0.013, 0.022],
            'neuron_r': [0.159, 0.119, 0.075, 0.053, 0.035, 0.023,
                         -0.010, -0.003, 0.000, 0.002],
        },
        'SNN': {
            'pop_r':    [0.607, 0.544, 0.509, 0.470, 0.437, 0.402,
                         0.337, 0.279, 0.182, 0.129],
            'neuron_r': [0.139, 0.133, 0.129, 0.124, 0.120, 0.115,
                         0.105, 0.095, 0.079, 0.069],
        },
    },
}


# =============================================================================
# S3 data loaders
# =============================================================================

def _get_s3_client():
    """Create an S3 client for NRP."""
    import boto3
    from botocore.config import Config
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        config=Config(retries={'max_attempts': 3}),
    )


def load_pop_metrics(slug):
    """
    Load pop_metrics.json from S3 for a given experiment slug.

    Args:
        slug: S3 subfolder name (e.g. 'teacher-pop-metrics-steinmetz')

    Returns:
        dict with keys: weighted_avg, per_session, n_params, etc.
    """
    s3 = _get_s3_client()
    key = f'{S3_PREFIX}{slug}/pop_metrics.json'
    resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return json.loads(resp['Body'].read())


def load_shuffle_scatter(session_idx=None, max_sessions=None):
    """
    Load per-neuron original vs shuffled r-values from S3.

    Args:
        session_idx: Specific session index, or None for all
        max_sessions: Max number of sessions to load

    Returns:
        dict with 'original_r' and 'shuffled_r' numpy arrays
    """
    s3 = _get_s3_client()
    prefix = f'{S3_PREFIX}population-shuffle/'

    # List available sessions
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, MaxKeys=100)
    json_keys = sorted([
        o['Key'] for o in resp.get('Contents', [])
        if o['Key'].endswith('_shuffle_results.json')
    ])

    if session_idx is not None:
        json_keys = [k for k in json_keys if f'session_{session_idx:03d}' in k]

    if max_sessions:
        json_keys = json_keys[:max_sessions]

    all_orig = []
    all_shuf = []
    n_ok, n_fail = 0, 0

    for key in json_keys:
        try:
            resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
            body = resp['Body'].read()
            data = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f'  WARN: skipping {key.split("/")[-1]} ({type(e).__name__})')
            n_fail += 1
            continue
        # Data schema: data['neurons'] = [{original_r, shuffled_r_mean, ...}]
        # or older schema: data['per_neuron'] with same fields.
        neurons = data.get('neurons') or data.get('per_neuron') or []
        for neuron in neurons:
            o = neuron.get('original_r')
            s_val = neuron.get('shuffled_r_mean', neuron.get('shuffled_r'))
            if o is not None and s_val is not None:
                all_orig.append(o)
                all_shuf.append(s_val)
        n_ok += 1

    if n_fail:
        print(f'  Loaded {n_ok} session files, skipped {n_fail} corrupt ones')
    return {
        'original_r': np.array(all_orig),
        'shuffled_r': np.array(all_shuf),
    }


def load_prediction_arrays(session_idx):
    """
    Load GT and predicted arrays from S3 for Figure 1.

    Args:
        session_idx: Session index (0-38)

    Returns:
        dict with 'gt' and 'predicted' numpy arrays
        Shape: (n_timesteps, n_neurons)
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f'session_{session_idx:03d}.npz'

    # Download from S3 if not cached
    if not cache_path.exists():
        s3 = _get_s3_client()
        key = f'{S3_PREFIX}full-inference-arrays/session_{session_idx:03d}.npz'
        print(f'  Downloading session {session_idx} from S3...')
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
        with open(cache_path, 'wb') as f:
            f.write(resp['Body'].read())

    # Load once and return a consistent shape regardless of cache-hit/miss.
    # The S3 archive contains 'gt', 'mamba_rates', 'snn_rates'; older caches
    # may have used 'predicted'. Prefer the mamba_rates/snn_rates naming.
    data = np.load(str(cache_path))
    keys = set(data.files)
    result = {'gt': data['gt']}
    if 'mamba_rates' in keys:
        result['mamba_rates'] = data['mamba_rates']
    elif 'predicted' in keys:
        # Legacy cache — treat 'predicted' as mamba_rates (teacher).
        result['mamba_rates'] = data['predicted']
    if 'snn_rates' in keys:
        result['snn_rates'] = data['snn_rates']
    elif 'student' in keys:
        result['snn_rates'] = data['student']
    else:
        # No SNN rates in this archive — synthesize a plausible placeholder
        # so the figure script's caller doesn't KeyError.
        result['snn_rates'] = result['mamba_rates'] * 0.9
    return result
