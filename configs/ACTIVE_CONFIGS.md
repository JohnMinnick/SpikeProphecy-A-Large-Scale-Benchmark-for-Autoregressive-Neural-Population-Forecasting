# Active Configs — SpikeProphecy

> These are the **only configs you should use** for new work.
> All stale/legacy configs have been moved to `configs/archive/`.

---

## Data
| Config | Purpose |
|--------|---------|
| `data/steinmetz_multi_nrp_50ms_no_cov.yaml` | **Production default** — 39 sessions, 50ms bins, no stimulus covariates |
| `data/steinmetz_multi_nrp_50ms_no_cov_t40.yaml` | Extended history (2s) variant |
| `data/steinmetz_multi_resolution.yaml` | Multi-resolution bin width sweeps |
| `data/overfit_sanity.yaml` | Single-session overfit sanity check |
| `data/ibl_repeated_site_50ms_no_cov.yaml` | IBL cross-dataset (paused) |
| `data/ibl_repeated_site_20_sessions_nrp.yaml` | IBL NRP variant (paused) |

## Teacher
| Config | Purpose |
|--------|---------|
| `teacher/nrp_teacher_mamba.yaml` | **Production Mamba teacher** |
| `teacher/nrp_teacher_mamba_session_heads.yaml` | Mamba + session-specific heads |
| `teacher/nrp_teacher_mamba_cmp.yaml` | Mamba comparison/ablation |
| `teacher/nrp_teacher_mamba_region_hybrid.yaml` | Region-weighted Mamba (experimental) |
| `teacher/overfit_sanity.yaml` | Single-session LSTM overfit test |

## Student
| Config | Purpose |
|--------|---------|
| `student/distill_mamba_multi_head.yaml` | **Production multi-head SNN distillation** |
| `student/distill_mamba_h256.yaml` | Standard 256-hidden distillation |
| `student/distill_nrp.yaml` | NRP distillation (standard, β=0.5) |
| `student/distill_beta0_ablation.yaml` | **β=0 KL ablation** — distill_weight=0.0, mechanistic test for NeurIPS §4.3 |
| `student/distill_gac_snn.yaml` | GAC-SNN mechanism alignment (experimental) |
| `student/standalone_snn.yaml` | Standalone SNN (no teacher) |
| `student/standalone_multihead_1l.yaml` | **1L SNN + stimulus/response heads** (neuromorphic twin) |
| `student/standalone_multihead_2l.yaml` | **2L SNN + stimulus/response heads** (neuromorphic twin) |
| `student/standalone_selective_h256.yaml` | SelectiveRSynaptic standalone |
| `student/default.yaml` | Base defaults |

## SHD
| Config | Purpose |
|--------|---------|
| `shd/mamba_shd.yaml` | Mamba on SHD classification |
| `shd/snn_shd.yaml` | SNN on SHD classification |
| `shd/snn_selective_shd.yaml` | Selective SNN on SHD |
