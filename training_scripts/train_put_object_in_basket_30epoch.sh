#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/modelscope/datasets/put_object_in_basket}"
OUT="${OUT:-outputs/smolvla_piper_put_object_in_basket_base_30epoch_bs72}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/root/autodl-tmp/huggingface/lerobot}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BATCH_SIZE="${BATCH_SIZE:-72}"
NUM_WORKERS="${NUM_WORKERS:-12}"
STEPS="${STEPS:-11773}"
SAVE_FREQ="${SAVE_FREQ:-1963}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

lerobot-train \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.load_vlm_weights=true \
  --policy.num_expert_layers=0 \
  --policy.prefix_length=0 \
  --policy.pad_language_to=max_length \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.use_amp=true \
  --policy.n_action_steps=50 \
  --policy.device=cuda \
  --policy.repo_id=piper-put-object-in-basket-smolvla-30epoch-bs72 \
  --policy.push_to_hub=false \
  --dataset.repo_id=yiyang2002/put_object_in_basket \
  --dataset.root="$DATA_ROOT" \
  --dataset.streaming=false \
  --dataset.video_backend="$VIDEO_BACKEND" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --log_freq=200 \
  --save_freq="$SAVE_FREQ" \
  --eval_freq=-1 \
  --wandb.enable=false \
  --output_dir="$OUT"
