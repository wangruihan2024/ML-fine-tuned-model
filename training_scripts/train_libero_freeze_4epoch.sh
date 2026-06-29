#!/usr/bin/env bash
set -euo pipefail

SNAP="${SNAP:-/root/autodl-tmp/huggingface/lerobot/hub/datasets--HuggingFaceVLA--libero/snapshots/86958911c0f959db2bbbdb107eb3e17c5f9c798e}"
OUT="${OUT:-outputs/smolvla_libero_freeze_4epoch_bs24_45580steps}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/root/autodl-tmp/huggingface/lerobot}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1

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
  --policy.n_action_steps=1 \
  --policy.device=cuda \
  --policy.repo_id=smolvla-libero-freeze-4epoch-bs24 \
  --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.root="$SNAP" \
  --dataset.streaming=false \
  --dataset.video_backend=torchcodec \
  --batch_size=24 \
  --num_workers=8 \
  --prefetch_factor=4 \
  --persistent_workers=true \
  --steps=45580 \
  --log_freq=100 \
  --save_checkpoint=true \
  --save_freq=45580 \
  --eval_freq=-1 \
  --use_policy_training_preset=true \
  --wandb.enable=false \
  --output_dir="$OUT"
