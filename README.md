# LookAway: Token-Level Counterfactual Voting for On-Policy Self-Distillation

LookAway is a token-level re-weighting framework for visual on-policy self-distillation, built on top of [Vision-OPD](https://arxiv.org/abs/2605.18740). Besides the standard student (full image) and positive teacher (evidence-centered crop) views, it constructs a negative teacher view — a same-format red box displaced to a wrong region — and uses the per-token score difference between the two teacher views to measure how much each token depends on seeing the right region. A four-gate eligibility funnel (view-dependence, word-rate prior, template bigram, JSD ammunition) selects answer-assertion tokens, and the distillation loss is re-weighted to concentrate extrapolation force on them.

This repository contains the training code, data preparation utilities, and evaluation scripts.

## Highlights

- Counterfactual three-view construction (student / positive teacher / negative teacher) with format-matched negative crops.
- Token-level view-dependence signal and a four-gate eligibility funnel for answer-assertion tokens.
- Frozen statistical priors (word selection rate + template bigrams) built once from a stage-0 dump.
- verl-based training stack with checkpoint merging, serving, and evaluation scripts.

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
export HF_CACHE_ROOT=./cache/huggingface
export DATA_DIR=./cache/lookaway
export WORK_DIR=./cache/lookaway
export WANDB_MODE=offline
```

`DATA_DIR/train.parquet` is the training file. Checkpoints and rollouts are written under `WORK_DIR`.

> **Critical (Qwen3.5 / `qwen3_next` models on CUDA-12.4 systems):** vLLM's flashinfer JIT-compiles the gated-delta-rule kernel with the system nvcc and fails when no CUDA-12.8 toolkit is installed. Force the triton GDN backend: add `+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.gdn_prefill_backend=triton` when training, and `--additional-config '{"gdn_prefill_backend":"triton"}'` when serving.

## Data

```
${PYTHON} scripts/prepare_data.py \
  --data-dir ./cache/lookaway \
  --hf-repo yuanqianhao/Vision-OPD-6K
```

This downloads the Vision-OPD-6K dataset, generates the negative teacher views (red box at a displaced position with IoU < 0.1 vs the ground-truth box, cropped and resized to match the official positive crop), and writes `./cache/lookaway/train.parquet` with the `neg_bbox_images` column.

## Priors

The voting gates rely on frozen statistical priors, built once from a stage-0 three-view dump of the base model:

```
CUDA_VISIBLE_DEVICES=0 \
${PYTHON} scripts/prepare_priors.py \
  --data-dir ./cache/lookaway \
  --model-path Qwen/Qwen3.5-4B
```

This greedily answers every training question from the student view, re-scores the answers under all three views using the same chat template as training, and derives the token priors (word selection rate + template bigrams) into `./cache/lookaway/token_priors.json`.

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
| `self_distillation.vd_dual_signal` | `True` | Use dynamic batch quantiles for the view-dependence and student-gap signals |
| `self_distillation.vd_base_signal` | `dd` | Base-weight signal: `dd` (view-dependence only) or `dual` (× teacher-student gap) |
| `self_distillation.vd_gamma` | `1.0` | Base-weight strength |
| `self_distillation.vd_tau` | `2.0` | Fixed-knee scale used when `vd_dual_signal=False` |
| `self_distillation.vd_targeted` | `True` | Enable the targeted extrapolation leg |
| `self_distillation.vd_lambda` | `4.0` | Extrapolation strength on gate-passing tokens |
| `self_distillation.vd_rate_cut` | `0.12` | Word-rate prior gate cut |
| `self_distillation.vd_jsd_gate` | `True` | JSD ammunition gate (per-token GJSD above batch median) |
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

Supported benchmarks: `vstar`, `zoombench`, `hrbench-4k`, `hrbench-8k`, `mme-realworld`, `mme-realworld-cn`, `mme-realworld-lite`, `visualprobe`, `mmvp`, `cv-bench`, `mmstar`, `pope`, `pope_adv`, `pope_pop`, and `pope_random`. For the Qwen3.5 baseline set `ENABLE_THINKING=False`. Thinking traces and common `Answer: X` wrappers are stripped before grading.

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

## License

Apache-2.0
