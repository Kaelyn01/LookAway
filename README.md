# LookAway: Token-Level Counterfactual Voting for On-Policy Self-Distillation

[![CI](https://github.com/Kaelyn01/LookAway/actions/workflows/ci.yml/badge.svg)](https://github.com/Kaelyn01/LookAway/actions/workflows/ci.yml)
[![Dataset](https://img.shields.io/badge/Hugging%20Face-CROP%20v2.0.0-FFD21E)](https://huggingface.co/datasets/haokaixinmeitiandouhaokaixin/crop-paired-visual-evidence)

LookAway is a token-level re-weighting framework for visual on-policy self-distillation, built on top of [Vision-OPD](https://arxiv.org/abs/2605.18740). Besides the standard student (full image) and positive teacher (evidence-centered crop) views, it constructs a negative teacher view — a same-format red box displaced to a wrong region — and uses the per-token score difference between the two teacher views to measure how much each token depends on seeing the right region. A four-gate eligibility funnel (view-dependence, word-rate prior, template bigram, JSD ammunition) selects answer-assertion tokens, and the distillation loss is re-weighted to concentrate extrapolation force on them.

This repository contains the training code, reproducible data preparation utilities, tests, and evaluation scripts. The aligned paired-image release is available on [Hugging Face](https://huggingface.co/datasets/haokaixinmeitiandouhaokaixin/crop-paired-visual-evidence) (`v2.0.0`; the earlier images remain under `v1.0.0`).

## Highlights

- Counterfactual three-view construction (student / positive teacher / negative teacher) with format-matched negative crops.
- Token-level view-dependence signal and a four-gate eligibility funnel for answer-assertion tokens.
- Frozen statistical priors (word selection rate + template bigrams) built once from a stage-0 dump.
- verl-based training stack with checkpoint merging, serving, and evaluation scripts.
- Pixel-exact reconstruction checks for all 6,241 official positive teacher images before aligned negatives are published.

## Environment

```
conda create -n lookaway python=3.12
conda activate lookaway

pip install --upgrade pip
pip install --no-deps -r requirements.txt
pip install -e . --no-deps
pip install flash-attn --no-build-isolation
pip install causal-conv1d==1.6.1 --no-build-isolation
```

Useful environment variables:

```
export PYTHON=python3
export HF_HOME=./cache/huggingface
export DATA_DIR=./cache/lookaway
export WORK_DIR=./cache/lookaway
export WANDB_MODE=offline
export MODEL_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

`DATA_DIR/train.parquet` is the training file. Checkpoints and rollouts are written under `WORK_DIR`.
The pinned model revision applies to the default `Qwen/Qwen3.5-4B`. Set both `MODEL_PATH` and its corresponding `MODEL_REVISION` when choosing another remote model.

> **Critical (Qwen3.5 / `qwen3_next` models on CUDA-12.4 systems):** vLLM's flashinfer JIT-compiles the gated-delta-rule kernel with the system nvcc and fails when no CUDA-12.8 toolkit is installed. Force the triton GDN backend: add `+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.gdn_prefill_backend=triton` when training, and `--additional-config '{"gdn_prefill_backend":"triton"}'` when serving.

## Data

```
${PYTHON} scripts/prepare_data.py \
  --data-dir ./cache/lookaway \
  --hf-repo yuanqianhao/Vision-OPD-6K \
  --hf-revision eb5c1c2e7b9a7b6a619efe4161c7369c71bf8af4
```

This downloads a pinned Vision-OPD-6K revision, reconstructs each official positive teacher image exactly, and generates a negative view by translating its complete crop template to a region with IoU < 0.1 against the target box. The original positive PNG is preserved. Positive and negative views share the recovered crop size, frame coordinates, Pillow 5-pixel stroke, and 2x LANCZOS resize. Generation happens in a staging directory and is published only after every row passes; a successful regeneration invalidates stale `train.parquet` and token priors before rebuilding them. The output `./cache/lookaway/train.parquet` contains `neg_bbox_images`, while `results.json` retains the complete geometry, hashes, and rendering parameters.

The public paired dataset documents all v2 geometry fields and validation results. Quality labels from v1 refer to the earlier negative images and remain historical until v2 receives a new semantic review.

To use the already prepared aligned image pairs without rebuilding them, load the [CROP v2.0.0 dataset](https://huggingface.co/datasets/haokaixinmeitiandouhaokaixin/crop-paired-visual-evidence). The training script expects the local Vision-OPD/LookAway Parquet schema produced by `prepare_data.py`; the public dataset is an auditable paired-image release rather than a drop-in replacement for `train.parquet`.

## Priors

The voting gates rely on frozen statistical priors, built once from a stage-0 three-view dump of the base model:

```
CUDA_VISIBLE_DEVICES=0 \
${PYTHON} scripts/prepare_priors.py \
  --data-dir ./cache/lookaway \
  --model-path Qwen/Qwen3.5-4B \
  --model-revision "$MODEL_REVISION"
```

This greedily answers every training question from the student view, re-scores the answers under all three views using the same chat template as training, and derives the token priors (word selection rate + template bigrams) into `./cache/lookaway/token_priors.json`. The file records the model revision, tokenizer fingerprint, template hash, negative-generation hash, and training-Parquet hash; the training launcher rejects stale or mismatched priors.

## Train

Default LookAway training (full method; one epoch over 6,241 samples = 65 steps at batch 96):

```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_PATH=Qwen/Qwen3.5-4B \
DATA_DIR=./cache/lookaway \
WORK_DIR=./cache/lookaway \
bash scripts/run_lookaway.sh
```

Two-step smoke test:

```
bash scripts/run_lookaway.sh trainer.total_training_steps=2
```

The unweighted distillation baseline (Vision-OPD recipe, no voting) is the vendored launcher directly:

```
bash scripts/run_vision_opd.sh
```

### Method Options

All method variants are verl config overrides; no code edits are needed. The
table shows the values set by `scripts/run_lookaway.sh` — the verl
config-file defaults leave every vd_* mechanism off, so the plain unweighted
recipe is `scripts/run_vision_opd.sh` without overrides.

| Flag | `run_lookaway.sh` value | Meaning |
| --- | --- | --- |
| `self_distillation.vd_dual_signal` | `True` | Use dynamic micro-batch quantiles for the view-dependence and student-gap signals |
| `self_distillation.vd_base_signal` | `dd` | Base-weight signal: `dd` (view-dependence only) or `dual` (× teacher-student gap) |
| `self_distillation.vd_gamma` | `1.0` | Base-weight strength |
| `self_distillation.vd_tau` | `2.0` | Fixed-knee scale used when `vd_dual_signal=False` |
| `self_distillation.vd_targeted` | `True` | Enable the targeted extrapolation leg |
| `self_distillation.vd_lambda` | `4.0` | Extrapolation strength on gate-passing tokens |
| `self_distillation.vd_rate_cut` | `0.12` | Word-rate prior gate cut |
| `self_distillation.vd_jsd_gate` | `True` | JSD ammunition gate (per-token GJSD above the current micro-batch median) |
| `self_distillation.vd_jsd_q` | `0.5` | JSD gate quantile |
| `self_distillation.vd_prior_file` | from `DATA_DIR` | Frozen prior table (see Priors) |

For example, the earlier fixed-knee single-signal variant (targeted
extrapolation and the JSD gate must be turned off together):

```
bash scripts/run_lookaway.sh \
  actor_rollout_ref.actor.self_distillation.vd_targeted=False \
  actor_rollout_ref.actor.self_distillation.vd_jsd_gate=False \
  actor_rollout_ref.actor.self_distillation.vd_dual_signal=False
```

## Merge Checkpoints

```
bash scripts/merge_checkpoint.sh ./cache/lookaway/checkpoints/<experiment>/global_step_65
```

The merged Hugging Face checkpoint is written back into the same `global_step_*` directory.

## Serve

```
vllm serve ./cache/lookaway/checkpoints/<experiment>/global_step_65 \
    --served-model-name LookAway-4B \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --port 8000 \
    --additional-config '{"gdn_prefill_backend":"triton"}'
```

## Evaluate

```
API_BASE=http://localhost:8000/v1/ \
OPENAI_MODEL_ID=LookAway-4B \
JUDGE_API_BASE=<judge-api-base> \
JUDGE_MODEL=<judge-model-name> \
BENCHMARK=vstar,zoombench \
bash eval/run_eval.sh
```

Supported benchmarks: `vstar`, `zoombench`, `hrbench-4k`, `hrbench-8k`, `mme-realworld`, `mme-realworld-cn`, `mme-realworld-lite`, `visualprobe`, `mmvp`, `cv-bench`, `mmstar`, `pope`, `pope_adv`, `pope_pop`, and `pope_random`. Their Hugging Face revisions are pinned in `eval/prepare_data.py`. For the Qwen3.5 baseline set `ENABLE_THINKING=False`. Thinking traces and common `Answer: X` wrappers are stripped before grading.

API credentials are read from `OPENAI_API_KEY` and `JUDGE_API_KEY`; the launcher does not place them in process arguments. Evaluation defaults to 32 concurrent requests and can be tuned with `PARALLEL_WORKERS`. Exhausted judge retries are recorded as errors and stop accuracy reporting instead of being counted as wrong answers.

## Repository layout

- `scripts/prepare_data.py`: pinned data download, exact positive reconstruction, aligned negative generation, and training Parquet export.
- `scripts/prepare_priors.py`: model-conditioned frozen prior construction.
- `scripts/run_lookaway.sh`: LookAway training entry point; validates data/prior compatibility first.
- `scripts/run_vision_opd.sh`: unweighted Vision-OPD baseline and shared training launcher.
- `eval/`: benchmark acquisition, inference, judging, and metrics.
- `tests/`: geometry, weighting, launcher, data-preparation, and evaluation regressions.
- `verl/`: vendored Vision-OPD/verl runtime with the LookAway integration.

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
ruff check scripts eval tests verl/workers/actor/lookaway_utils.py
ruff check --select F verl/workers/actor/dp_actor.py verl/trainer/ppo/ray_trainer.py verl/workers/config/actor.py
bash -n scripts/*.sh eval/*.sh
```

The default Qwen model is resolved to the pinned revision in `MODEL_REVISION` before training. If you intentionally use another remote model, set its corresponding revision. For a trusted local model directory, set `MODEL_PATH` to that directory; the revision is ignored. Hugging Face model loading uses `trust_remote_code=True`, so audit code when changing the model source.

## Reproducibility notes

- The repository vendors the Vision-OPD/verl runtime from base revision `c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471`; project-specific changes are identified in the Git history.
- Training data, the default model, and evaluation datasets are revision-pinned. Override a remote source only together with its revision.
- Dynamic p90 thresholds are computed over valid tokens in each training micro-batch. Results can therefore depend on sequence packing, device count, and batch order; record the launch configuration and seed.
- `requirements.txt` is the reproduced full environment lock rather than a minimal abstract dependency list. CUDA/PyTorch wheels remain platform-specific.

## Citation

This repository builds on Vision-OPD; if you use it, please cite the original work:

```
@article{yuan2026vision,
  title={Vision-OPD: Learning to See Fine-Grained Details for Multimodal LLMs via On-Policy Self-Distillation},
  author={Yuan, Qianhao and Lou, Jie and Yu, Xing and Lin, Hongyu and Sun, Le and Han, Xianpei and Lu, Yaojie},
  journal={arXiv preprint arXiv:2605.18740},
  year={2026}
}
```

## License and provenance

Repository code is Apache-2.0; see `LICENSE` and `NOTICE`. The training data is a derived resource with separate upstream image terms. Consult the [CROP dataset card](https://huggingface.co/datasets/haokaixinmeitiandouhaokaixin/crop-paired-visual-evidence) before redistributing or using the images.
