#!/bin/bash
# LookAway training launcher (full method by default).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 \
#   MODEL_PATH=Qwen/Qwen3.5-4B \
#   MODEL_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
#   DATA_DIR=./cache/lookaway \
#   WORK_DIR=./cache/lookaway \
#   bash scripts/run_lookaway.sh
#
# Extra hydra overrides pass through, e.g. a 2-step smoke test:
#   bash scripts/run_lookaway.sh trainer.total_training_steps=2
#
# The unweighted distillation baseline is scripts/run_vision_opd.sh directly
# (the vendored official launcher, without the vd_* overrides).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/cache/lookaway}"
WORK_DIR="${WORK_DIR:-$PROJECT_ROOT/cache/lookaway}"
DEFAULT_MODEL_PATH="Qwen/Qwen3.5-4B"
DEFAULT_MODEL_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
MODEL_REVISION="${MODEL_REVISION:-}"
if [[ "$MODEL_PATH" == "$DEFAULT_MODEL_PATH" && -z "$MODEL_REVISION" ]]; then
  MODEL_REVISION="$DEFAULT_MODEL_REVISION"
fi
MODEL_NAME="$(basename "$MODEL_PATH")"
TRAIN_FILE="$DATA_DIR/train.parquet"
PRIOR_FILE="$DATA_DIR/token_priors.json"
FREQ_FILE="${VD_FREQ_FILE:-$DATA_DIR/token_freq.json}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-LookAway-${MODEL_NAME}}"
PROJECT_NAME="${PROJECT_NAME:-LookAway}"
export EXPERIMENT_NAME PROJECT_NAME MODEL_PATH MODEL_REVISION

for f in "$TRAIN_FILE" "$PRIOR_FILE" "$DATA_DIR/results.json"; do
  if [ ! -f "$f" ]; then
    echo "Missing $f -- run scripts/prepare_data.py and scripts/prepare_priors.py first." >&2
    exit 1
  fi
done

"${PYTHON:-python3}" "$SCRIPT_DIR/validate_priors.py" \
  --prior-file "$PRIOR_FILE" \
  --model-path "$MODEL_PATH" \
  --model-revision "$MODEL_REVISION" \
  --chat-template "$PROJECT_ROOT/chat_templates/perception_chat_template_qwen35.jinja" \
  --generation-results "$DATA_DIR/results.json" \
  --train-file "$TRAIN_FILE"

# Frequency decay (budget-matched extrapolation reallocation). Enabled by
# default; disable with VD_FREQ_DECAY=false to train without reallocation.
FD_OVERRIDES=()
if [[ "${VD_FREQ_DECAY:-true}" == "true" ]]; then
  if [ ! -f "$FREQ_FILE" ]; then
    echo "Missing $FREQ_FILE -- run scripts/build_freq_table.py first (or set VD_FREQ_DECAY=false)." >&2
    exit 1
  fi
  FD_OVERRIDES=(
    +actor_rollout_ref.actor.self_distillation.vd_freq_decay=true
    +actor_rollout_ref.actor.self_distillation.vd_freq_file="$FREQ_FILE"
  )
fi

exec bash "$SCRIPT_DIR/run_vision_opd.sh" \
  data.train_files="[\"$TRAIN_FILE\"]" \
  trainer.n_gpus_per_node="${TRAINER_N_GPUS_PER_NODE:-4}" \
  trainer.total_epochs=1 \
  +actor_rollout_ref.actor.self_distillation.teacher_neg_image_key=neg_bbox_images \
  +actor_rollout_ref.actor.self_distillation.vd_gamma=1.0 \
  +actor_rollout_ref.actor.self_distillation.vd_tau=2.0 \
  +actor_rollout_ref.actor.self_distillation.vd_dual_signal=true \
  +actor_rollout_ref.actor.self_distillation.vd_base_signal=dd \
  +actor_rollout_ref.actor.self_distillation.vd_targeted=true \
  +actor_rollout_ref.actor.self_distillation.vd_lambda=4.0 \
  +actor_rollout_ref.actor.self_distillation.vd_rate_cut=0.12 \
  +actor_rollout_ref.actor.self_distillation.vd_prior_file="$PRIOR_FILE" \
  +actor_rollout_ref.actor.self_distillation.vd_jsd_gate=true \
  +actor_rollout_ref.actor.self_distillation.vd_jsd_q=0.5 \
  ${FD_OVERRIDES[@]+"${FD_OVERRIDES[@]}"} \
  trainer.default_local_dir="$WORK_DIR/checkpoints/$EXPERIMENT_NAME" \
  trainer.rollout_data_dir="$WORK_DIR/rollouts/$EXPERIMENT_NAME" \
  "$@"
