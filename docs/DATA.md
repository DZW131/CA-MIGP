# Data preparation

The repository does not redistribute images, annotations, or derived model artifacts.
Review and follow each provider's terms before downloading a dataset.

## PanNuke

- Provider page: https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke
- Public parquet mirror used by the preparation script:
  https://huggingface.co/datasets/MedOtter/PanNuke
- Frozen roles: fold 1 training, fold 2 validation/threshold selection, fold 3 testing.

Prepare all downloaded parquet shards:

```bash
python scripts/prepare_pannuke.py \
  --input data/raw/pannuke/fold1/*.parquet \
          data/raw/pannuke/fold2/*.parquet \
          data/raw/pannuke/fold3/*.parquet \
  --output-root data/processed/pannuke \
  --manifest data/manifests/pannuke.jsonl
```

Every image produces one center-detection record and one foreground-segmentation record.
The instance map is retained for geometry-target construction and instance evaluation.

## NuInsSeg

- Release: https://doi.org/10.5281/zenodo.10518968
- Role: frozen external evaluation.

```bash
python scripts/prepare_nuinsseg.py \
  --source data/raw/NuInsSeg \
  --output-root data/processed/nuinsseg \
  --manifest data/manifests/nuinsseg.jsonl
```

The script uses the modified label masks and rasterizes the provided vague-area ImageJ ROIs
as ignore regions. The resulting split is `ood_test`.

## CryoNuSeg

- Repository: https://github.com/masih4/CryoNuSeg
- Primary reference: Annotator 1, round 1 modified masks.
- Sensitivity references: Annotator 1, round 2 and Annotator 2.

```bash
python scripts/prepare_cryonuseg.py \
  --archive data/raw/cryonuseg/cryonuseg-kaggle.zip \
  --source data/raw/cryonuseg/extracted \
  --output-root data/processed/cryonuseg \
  --manifest-dir data/manifests/cryonuseg
```

The source directory must contain the extracted provider release. All generated manifests
use `ood_test` and must remain excluded from checkpoint, threshold, and post-processing
selection.

## Manifest fields

Each JSONL record contains an image path, label path, operation, target, domain, grouping
identifier, and split. Optional fields carry ignore masks and annotation metadata. Paths may
be absolute or relative to the process working directory.
