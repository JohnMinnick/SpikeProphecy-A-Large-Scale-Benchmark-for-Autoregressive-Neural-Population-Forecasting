# For NeurIPS 2026 reviewers

This file is the fast path for E&D track reviewers who want to spot-check
the benchmark without spinning up the full pipeline.

## What's where

* **Paper (compiled PDF):** `docs/paper/main.pdf`
* **Paper source:** `docs/paper/main.tex`
* **NeurIPS checklist:** `docs/paper/checklist.tex`
* **Datasheet (Appendix B excerpt of `main.pdf`):** `docs/paper/main.tex` line ~1719 onward

## Three quick spot-checks

### 1. The leakage-audit suite is real

The paper claims a 14-test audit suite covering 5 leakage vectors,
and that the audit catches the Population-GLM result reported as a
canonical example.

```bash
pytest tests/test_data/                # full data-side audit
pytest tests/test_eval/test_glm_baseline.py   # PopGLM leakage catch
```

### 2. Population metric definitions match the paper

The three component metrics in the paper (Eqs. 2-4 of the
benchmark-design section) are:

* `pop_rate_r`     - temporal fidelity (cross-time Pearson on
                     population-summed rates)
* `spatial_r`      - spatial fidelity (per-bin cross-neuron Pearson,
                     then averaged over time)
* `cosine_sim`     - magnitude-invariant alignment

Implemented in `src/eval/population_metrics.py`:

```bash
pytest tests/test_eval/test_population_metrics.py
```

### 3. The seven architecture factories all instantiate

```bash
pytest tests/test_models/   # exercises Mamba, HGRN2, GatedDelta,
                            # Transformer, LRU, LSTM, SNN factories
```

## Reproducing a single row of Table 3

A single Steinmetz-39 row of the main results table is reproducible
on a workstation GPU with the relevant teacher config:

```bash
python scripts/nrp_train.py \
    --data-config configs/data/steinmetz_multi_nrp_50ms_no_cov.yaml \
    --teacher-config configs/teacher/nrp_teacher_mamba.yaml \
    --epochs 50 --seed 42

python scripts/eval_local_teacher.py \
    --checkpoint <path-to-best.pt> \
    --output-json mamba_metrics.json
```

The full Table 3 used a single GPU run per architecture per seed.
Three-seed estimates (Mamba, HGRN2, Transformer) were run on the
National Research Platform; remaining four architectures
(GatedDeltaNet, LRU, LSTM, SNN) are single-seed.

## Reproducing the figures

```bash
cd scripts/figures
python figure1_hero_v6.py        # Fig 1
python figure3_findings.py       # Fig 2 (a) + (b)
python figure_appendix.py        # appendix figures (AR rollout etc.)
```

Cached intermediate stats (used to avoid re-running NRP jobs at
figure time) live in `data/figure_cache/` -- not in this repo
because they are post-evaluation artifacts. The figure scripts will
recompute from raw model outputs if the cache is absent.

## Source datasets

We do not redistribute the raw recordings. Both source datasets
are publicly hosted under CC-BY-4.0:

* Steinmetz 2019 -> https://doi.org/10.6084/m9.figshare.9598406.v2
* IBL Repeated Site -> https://www.internationalbrainlab.com/data

Our `scripts/run_ibl_cache.py` and `scripts/build_combined_cache.py`
fetch the source data and produce the 50 ms-binned tensors used by
everything downstream.

## Anonymization note

Author-identifying material has been removed from this repo for
double-blind review. Commit history before this submission lives
in a private working repo and will be merged at acceptance.
