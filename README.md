# CA-MIGP

Official implementation of **Contact-Aware Instance Geometry Prototypes with Dynamic
Gram-Style Prompting for Source-Only Cross-Dataset Nucleus Segmentation**.

CA-MIGP is a training-only geometry objective for compact nucleus instance segmentation.
It decomposes each annotated nucleus into scale-normalized core, shell, free-rim, and
contact-rim states, adds a near-background state, and aligns sampled decoder features with
cross-tissue exponential-moving-average prototypes. The complete model combines this local
geometry objective with an auxiliary Dynamic Gram-style appearance prompt. At inference,
the geometry projector and prototype bank are inactive.

## Method components

- **Dynamic Gram:** image-conditioned channel and spatial modulation plus a dynamic output head.
- **CA-MIGP:** five instance-scale geometry states with state-balanced sampling and EMA prototypes.
- **Contact awareness:** contact-rim pixels are identified using neighboring instance identities.
- **Instance reconstruction:** center heatmaps and nuclear foreground are combined by seeded watershed.

## Repository layout

```text
configs/   Paper configurations, controls, and a small smoke configuration
docs/      Dataset and reproducibility notes
scripts/   Data preparation, training, evaluation, statistics, and visualization
src/pathm/ Model, geometry targets, losses, metrics, and post-processing
tests/     Unit and configuration-pairing tests
```

## Installation

Python 3.10 or 3.11 is recommended. Create an isolated environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For NVIDIA GPUs, install the PyTorch build matching the local CUDA driver before the
editable installation. The reported experiments used Python 3.10.20, PyTorch 2.11.0,
CUDA 12.8, cuDNN 9.1.9, NumPy 2.2.6, SciPy 1.15.3, and scikit-image 0.25.2.

## Data

Datasets are not redistributed. Download them from their providers and prepare JSONL
manifests as described in [docs/DATA.md](docs/DATA.md). The expected default location is:

```text
data/
  raw/
  processed/
  manifests/
    pannuke.jsonl
    nuinsseg.jsonl
    cryonuseg/
```

Validate a prepared manifest with:

```bash
python scripts/validate_manifest.py data/manifests/pannuke.jsonl
```

## Paper configurations

The public files retain the run names used by the experiment artifacts. Only machine-local
output and manifest paths were changed to repository-relative paths.

| Cell/control | Mechanisms | Configuration family |
|---|---|---|
| A: Base | neither mechanism | `pannuke_ablation_no_style_seed{17,23,42}.yaml` |
| B: Dynamic Gram | Dynamic Gram only | `pannuke_full_seed{17,23,42}.yaml` |
| C: CA-MIGP | CA-MIGP only | `pannuke_no_style_geometry_seed{17,23,42}.yaml` |
| D: Complete | Dynamic Gram + CA-MIGP | `pannuke_geometry_no_orth_seed{17,23,42}.yaml` |
| Matched no-contact | D without the explicit contact package | `pannuke_camigp_no_contact_seed{17,23,42}.yaml` |
| Direct GAP | fixed-width natural-image transfer control | `pannuke_gap_control_seed{17,23,42}.yaml` |
| Style-orthogonal | rejected Gram-direction subtraction | `pannuke_somigp_seed{17,23,42}.yaml` |
| Constant direction | seed-17 localization control | `pannuke_geometry_constant_direction_seed17.yaml` |

The legacy no-contact plus style-projection configurations are retained as
`pannuke_geometry_no_contact_seed{17,23,42}.yaml` for supplementary provenance only.

## Smoke test

After preparing a PanNuke manifest, run a one-epoch, eight-record smoke test:

```bash
python scripts/smoke_test.py --config configs/smoke_complete.yaml
```

CPU execution is supported for tests and smoke checks. Full training is intended for CUDA.

## Training

Train one complete-model seed:

```bash
python scripts/train.py \
  --config configs/pannuke_geometry_no_orth_seed17.yaml \
  --device cuda
```

The run directory is `outputs/<run_name>/` and contains `config.yaml`, `best.pt`, `last.pt`,
and `summary.json`. Repeat with seeds 23 and 42 for the paired three-seed design.

## Threshold selection

Detection and segmentation thresholds are selected on PanNuke fold 2 and then frozen:

```bash
python scripts/tune_threshold.py \
  --config configs/pannuke_geometry_no_orth_seed17.yaml \
  --checkpoint outputs/pannuke_geometry_no_orth_seed17/best.pt \
  --task detection \
  --values 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.50 0.60 0.70 0.80 \
  --output outputs/pannuke_geometry_no_orth_seed17/detection_threshold.json

python scripts/tune_threshold.py \
  --config configs/pannuke_geometry_no_orth_seed17.yaml \
  --checkpoint outputs/pannuke_geometry_no_orth_seed17/best.pt \
  --task segmentation \
  --values 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 \
  --output outputs/pannuke_geometry_no_orth_seed17/segmentation_threshold.json
```

## Instance evaluation

Evaluate a frozen checkpoint on an external manifest without target-domain tuning:

```bash
python scripts/evaluate_instances.py \
  --config configs/pannuke_geometry_no_orth_seed17.yaml \
  --checkpoint outputs/pannuke_geometry_no_orth_seed17/best.pt \
  --manifest data/manifests/nuinsseg.jsonl \
  --split ood_test \
  --detection-threshold 0.10 \
  --segmentation-threshold 0.50 \
  --output-dir outputs/pannuke_geometry_no_orth_seed17/nuinsseg_ood
```

Replace the example thresholds with the two values selected on fold 2. The evaluator writes
task metrics, instance summaries, per-image records, prediction maps, and contact/boundary
diagnostics.

## Bootstrap analysis

The paired multi-seed and four-cell factorial utilities reproduce the hierarchical
resampling design used in the manuscript. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
for required artifact layouts and complete commands.

```bash
python scripts/bootstrap_gram_camigp_factorial.py --help
python scripts/bootstrap_compare_multiseed.py --help
```

## Visualization and efficiency

```bash
python scripts/visualize_predictions.py \
  --config configs/pannuke_geometry_no_orth_seed17.yaml \
  --checkpoint outputs/pannuke_geometry_no_orth_seed17/best.pt \
  --manifest data/manifests/nuinsseg.jsonl \
  --split ood_test \
  --output-dir outputs/visualizations \
  --max-images 8

python scripts/benchmark_model.py \
  --config configs/pannuke_geometry_no_orth_seed17.yaml \
  --checkpoint outputs/pannuke_geometry_no_orth_seed17/best.pt \
  --output outputs/benchmark_seed17.json \
  --device cuda
```

## Reported results

Three-seed means from the manuscript are shown for orientation. Recomputed values depend on
the downloaded dataset releases, software stack, and nondeterministic CUDA execution.

| Frozen test | Dynamic Gram (B) mean PQ | Complete (D) mean PQ | D minus B |
|---|---:|---:|---:|
| PanNuke fold 3 | 0.559 | 0.561 | 0.0018 |
| NuInsSeg | 0.332 | 0.335 | 0.0031 |
| CryoNuSeg A1R1 | 0.433 | 0.442 | 0.0083 |

The complete checkpoint contains 2.58 million parameters, while its active inference graph
contains 2.56 million parameters and requires 43.36 GFLOPs for a 256 x 256 image.

## Tests

```bash
pytest
ruff check src scripts tests
```

## Citation

The manuscript is under journal submission. Citation metadata are provided in
[`CITATION.cff`](CITATION.cff) and will be updated with the final DOI after publication.

## License

Code is released under the [MIT License](LICENSE). Dataset licenses remain with their
original providers.
