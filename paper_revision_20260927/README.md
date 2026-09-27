# Five-seed revision release

This directory records the A--D factorial analysis used in the revised *Electronics*
manuscript. The 20 runs pair seeds 17, 23, 42, 101, and 202 across A (neither
component), B (Dynamic Gram), C (CA-MIGP), and D (both). PanNuke fold 1 supplied
training labels; fold 2 selected checkpoints and thresholds; fold 3, NuInsSeg,
and CryoNuSeg were evaluated with the selected settings. Earlier external results
influenced method development, so the latter two datasets are **retrospective**
external evaluations rather than independent confirmation.

## Files

- `frozen_python_source.tar.gz`: Python source files from the archived training
  revision, including the training, model, geometry, metric, and evaluation code.
  This exact-source snapshot is separate from the repository's current `src/`,
  which contains later extensions. The ten training-source files in the archived
  fingerprint match this snapshot byte-for-byte.
- `configs/`: portable copies of the 20 frozen run configurations. Only the
  machine-specific output and PanNuke manifest paths were replaced by relative
  paths. Run names, seeds, model switches, optimizer settings, and evaluation
  settings are unchanged.
- `run_provenance.csv`: original frozen-configuration and checkpoint SHA-256
  digests plus SHA-256 digests of the portable configurations. The latter differ
  because of the path substitution.
- `all_run_summary_metrics.csv`: one row per cell, seed, and dataset, with the
  final frozen summary metrics.
- `five_seed_factorial_effects.csv`: exported paired contrasts for each dataset.
- `analyze_five_seed.py`: independently recomputes the mean-PQ contrasts and
  paired 95% $t$ intervals from `all_run_summary_metrics.csv`.

Run the summary analysis with `python analyze_five_seed.py` after installing
SciPy. To inspect or use the exact archived Python code, unpack
`frozen_python_source.tar.gz` into a separate directory. Install its
`pyproject.toml` requirements, provide the datasets according to the main
repository's data documentation, and use the corresponding portable YAML file.
Model checkpoints and derived per-image metrics are not included in this small
release; the manuscript identifies how to request them from the corresponding
author. The aggregate CSVs reproduce the manuscript's five-seed estimates but
are not a substitute for its original per-image prediction artifacts.

The source archive SHA-256 is
`4ef3c061c688dff0b96a491da6fe6d79fc598d44e720a51103ece73d741edfb0`.
