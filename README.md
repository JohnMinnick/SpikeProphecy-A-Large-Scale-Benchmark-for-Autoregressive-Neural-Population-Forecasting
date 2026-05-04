# SpikeProphecy

> *A Large-Scale Benchmark for Autoregressive Neural Population Forecasting*
>
> Companion code for the NeurIPS 2026 Evaluations & Datasets track submission.

The paper introduces SpikeProphecy: a 105-session, ~89,800-neuron
benchmark for causal, autoregressive spike-count forecasting on real
electrophysiology recordings, plus a population metric decomposition
(temporal fidelity, spatial pattern accuracy, magnitude-invariant
alignment) and seven matched architecture baselines (Mamba, HGRN2,
GatedDeltaNet, LRU, Transformer, LSTM, RSynaptic SNN).

The paper PDF lives at `docs/paper/main.pdf`.

## Repo layout

```
src/                  Library code, importable as `src.*`
  data/               Spike binning, dataset wrappers, leakage suite
  models/             Seven architecture factories
  distill/            ANN->SNN distillation utilities (Appendix C)
  eval/               Population metric decomposition + ceilings
  train/              Training loops
  utils/              Config, device, seeding helpers
  viz/                Plotting helpers (figure-pipeline-internal)
scripts/              Entry points: training, evaluation, inference
  figures/            Figure-generation pipeline (one script per panel)
configs/              YAML configs for data, teachers, students
  data/               Steinmetz / IBL / combined-105 dataset configs
  teacher/            Per-architecture teacher (full-size) configs
  student/            SNN student / distillation configs
tests/                pytest suite (~64 files; leakage tests at
                      `tests/test_data/test_*.py`)
docs/paper/           main.tex + main.pdf + figures + checklist +
                      OpenReview submission folder
Dockerfile.nrp        Docker image used for the National Research
                      Platform (Nautilus K8s) cluster training runs
pyproject.toml        Package metadata; install editable
```

## Data

The benchmark runs on two **public** datasets, both used unmodified
at the spike-time level. We do not redistribute the source recordings;
we redistribute only our processed 50 ms-binned tensors plus the
preprocessing code that derives them.

| Dataset | Size | Source | License |
|---|---|---|---|
| Steinmetz 2019 | 39 sessions, 27,212 neurons | [Figshare DOI](https://doi.org/10.6084/m9.figshare.9598406.v2) (raw) / [HuggingFace](https://huggingface.co/datasets/mysteriousauthor/spikeprophecy-steinmetz) (our processed tensors) | CC-BY-4.0 |
| IBL Repeated Site | 66 sessions, ~62,500 neurons | [IBL ONE API](https://www.internationalbrainlab.com/data) | CC-BY-4.0 |

`scripts/run_ibl_cache.py` and `scripts/build_combined_cache.py`
download and bin the source data into the format used by the rest
of the pipeline. See `configs/data/*.yaml` for the standard splits.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv/Scripts/activate on Windows
pip install -e ".[dev,nrp]"
```

The package import name is `src` (the project is `spike-prophecy`
on PyPI but ships under the `src/` namespace), so e.g.

```python
from src.models.mamba_baseline import MambaBaseline
from src.eval.population_metrics import compute_population_metrics
```

## Reproducing key results

These scripts reproduce the main paper tables and figures, in the
order a reviewer is most likely to want them:

| Table / Figure | Script |
|---|---|
| Tab 3 (main results, 7 archs on Steinmetz) | `scripts/eval_local_teacher.py` per arch |
| Tab 4 / Fig 2(a) (per-region predictability) | `scripts/eval_region_hierarchy_residualized.py` |
| Fig 2(b) (Fano-stratified per-neuron r) | `scripts/figures/_compute_fano_7arch_39session.py` |
| Tab 7 (efficiency: throughput, latency, VRAM) | `scripts/bench_inference_efficiency.py` |
| Fig 1 (hero) | `scripts/figures/figure1_hero_v6.py` |
| Fig 2 (findings) | `scripts/figures/figure3_findings.py` |
| App C SNN distillation table | `scripts/eval_distill_posthoc.py` + `scripts/collate_distillation_table.py` |
| App leakage audit | `pytest tests/test_data/test_*leakage*.py` |

Population metric primitives (`pop_rate_r`, `spatial_r`,
`cosine_sim`) are at `src/eval/population_metrics.py`.

## Tests

```bash
pytest                     # full suite (64 files)
pytest tests/test_eval     # population-metric unit tests
pytest tests/test_data     # leakage audit suite (the canonical
                           # PopGLM-as-leakage-catch test lives here)
```

## License

Code: MIT (see `LICENSE`).
Processed data + paper: CC-BY-4.0.
Source recordings retain their original CC-BY-4.0 licenses at the
hosts listed above.

## Citation

```bibtex
@inproceedings{spikeprophecy2026,
  title  = {SpikeProphecy: A Large-Scale Benchmark for
            Autoregressive Neural Population Forecasting},
  author = {Anonymous},
  booktitle = {NeurIPS 2026 Datasets and Benchmarks Track},
  year   = {2026}
}
```
