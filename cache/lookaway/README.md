# Frozen training artifacts

These files are the exact inputs consumed by the `lookaway-final` training run
(60 steps, one epoch over 5,795 samples). They are versioned deliberately:
`teacher_posneg_token_scores.jsonl` and the derived frozen priors are
experimental records that cannot be regenerated bit-exactly.

| file | role |
|---|---|
| `trainset_clean.parquet` / `.jsonl` | cleaned training set (questions + image paths + answers + geometry) |
| `dataset_generation_manifest.json` | per-sample build manifest (boxes, hashes, alignment checks) |
| `teacher_posneg_token_scores.jsonl` | per-token teacher scoring under positive/negative views (original record) |
| `token_priors_frozen.json` | frozen virtual-teacher priors consumed by training |
| `token_freq_counts.json` | token frequency table for frequency-decay reallocation |
| `rollouts/lookaway-sample-level-60steps/` | per-step rollout record of the final run |

## Provenance and license

The training set is derived from
[Vision-OPD-6K](https://huggingface.co/datasets/yuanqianhao/Vision-OPD-6K)
(question text and geometry; images are NOT redistributed here).
Redistribution follows the component-specific terms of the
[CROP release](https://huggingface.co/datasets/Kaelyn01/crop-paired-visual-evidence).
Regenerate everything with `scripts/import_crop_release.py` / `scripts/build_trainset.py` /
`scripts/prepare_priors.py` / `scripts/build_freq_table.py`.
