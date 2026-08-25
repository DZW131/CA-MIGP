# Reproducibility protocol

## Frozen experimental roles

1. Train on PanNuke fold 1.
2. Select checkpoints and thresholds on PanNuke fold 2.
3. Evaluate once on PanNuke fold 3, NuInsSeg, and CryoNuSeg with frozen settings.
4. Use paired optimization seeds 17, 23, and 42 for every A/B/C/D cell.

No target-domain image is used to fit network parameters, normalization statistics,
thresholds, or post-processing settings.

## Factorial cells

| Cell | Dynamic Gram | CA-MIGP | Config stem |
|---|---:|---:|---|
| A | off | off | `pannuke_ablation_no_style_seed` |
| B | on | off | `pannuke_full_seed` |
| C | off | on | `pannuke_no_style_geometry_seed` |
| D | on | on | `pannuke_geometry_no_orth_seed` |

All configurations use the same backbone, optimizer, data records, augmentation, 50-epoch
budget, and reconstruction settings. The `geometry_style_direction_mode: none` field in C
and D is intentional: Gram-direction subtraction was rejected by the negative-control
experiments and is not part of the retained method.

## Three-seed training

```bash
for seed in 17 23 42; do
  python scripts/train.py \
    --config configs/pannuke_geometry_no_orth_seed${seed}.yaml \
    --device cuda
done
```

Run the same loop for the A, B, and C configuration stems. CUDA was seeded but not forced
into bit-deterministic mode; retain all per-seed outputs rather than expecting byte-identical
checkpoints.

## Expected artifact layout

```text
outputs/
  pannuke_geometry_no_orth_seed17/
    best.pt
    last.pt
    config.yaml
    summary.json
    pannuke_fold3/
      per_image_tasks.jsonl
      test_instances.jsonl
    nuinsseg_ood/
      per_image_tasks.jsonl
      ood_test_instances.jsonl
    cryonuseg_ood/
      per_image_tasks.jsonl
      ood_test_instances.jsonl
```

The evaluation command controls the output directory, so equivalent relative layouts are
acceptable as long as the same layout is used for every paired seed and factorial cell.

## Paired bootstrap

For a candidate/reference comparison on NuInsSeg:

```bash
python scripts/bootstrap_compare_multiseed.py \
  --outputs-root outputs \
  --candidate-stem pannuke_geometry_no_orth_seed \
  --reference-stem pannuke_full_prompt_seed \
  --relative-dir nuinsseg_ood \
  --task-file per_image_tasks.jsonl \
  --instance-file ood_test_instances.jsonl \
  --cluster-key domain \
  --seeds 17 23 42 \
  --replicates 10000 \
  --seed 2026 \
  --output outputs/bootstrap_complete_minus_gram_nuinsseg.json
```

The run-name stem for B is `pannuke_full_prompt_seed`, although its configuration filename
is `pannuke_full_seed<seed>.yaml`.

## Four-cell factorial bootstrap

The factorial utility resamples seeds, datasets, and dataset-specific groups:

```bash
python scripts/bootstrap_gram_camigp_factorial.py \
  --outputs-root outputs \
  --dataset-spec pannuke_fold3:sample_id:per_image_tasks.jsonl:test_instances.jsonl \
  --dataset-spec nuinsseg_ood:domain:per_image_tasks.jsonl:ood_test_instances.jsonl \
  --dataset-spec cryonuseg_ood:domain:per_image_tasks.jsonl:ood_test_instances.jsonl \
  --seeds 17 23 42 \
  --replicates 10000 \
  --seed 2026 \
  --output outputs/factorial_bootstrap.json
```

Use `python scripts/bootstrap_gram_camigp_factorial.py --help` to override any run stem or
artifact location. The helper filename `bootstrap_gram_gsc_factorial.py` is retained for
compatibility with the locked experiment artifacts; the public CA-MIGP entry point is
`bootstrap_gram_camigp_factorial.py`.

## Integrity checks

```bash
pytest
ruff check src scripts tests
rg -n "/home/|10\\.15\\.|password|passwd|token|secret" .
```

Before reporting results, verify that paired cells use identical data manifests and that
the only configuration differences are the declared Dynamic Gram and CA-MIGP switches.
