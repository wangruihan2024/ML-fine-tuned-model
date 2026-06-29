#!/usr/bin/env bash
set -u

EXP_DIR="/root/autodl-tmp/lerobot/outputs/libero_bs_ablation_20260627_short1500_configpath"
CONFIG_PATH="/root/autodl-tmp/lerobot/outputs/smolvla_libero_base_direct_5epoch_bs24_56975steps_explicit_20260520_2225utc/checkpoints/045580/pretrained_model/train_config.json"
SNAP="/root/autodl-tmp/huggingface/lerobot/hub/datasets--HuggingFaceVLA--libero/snapshots/86958911c0f959db2bbbdb107eb3e17c5f9c798e"
BATCHES=(16 24 32 48)
TOTAL_STEPS=1500
LOG_FREQ=50

mkdir -p "${EXP_DIR}"

cat > "${EXP_DIR}/experiment_config.json" <<EOF
{
  "task": "LIBERO SmolVLA batch size ablation",
  "date_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "config_path": "${CONFIG_PATH}",
  "base_policy_pretrained_path": "lerobot/smolvla_base",
  "dataset_repo_id": "HuggingFaceVLA/libero",
  "dataset_root": "${SNAP}",
  "batch_sizes": [16, 24, 32, 48],
  "total_steps": ${TOTAL_STEPS},
  "warmup_steps": 500,
  "measurement_steps": 1000,
  "log_freq": ${LOG_FREQ},
  "num_workers": 8,
  "prefetch_factor": 4,
  "save_checkpoint": false,
  "eval_freq": -1,
  "notes": "Uses the previous successful LIBERO train_config.json so the LIBERO two-image/state8/action7 schema is preserved. Only batch_size changes across runs."
}
EOF

for BS in "${BATCHES[@]}"; do
  RUN_DIR="${EXP_DIR}/bs${BS}"
  TRAIN_OUT="${RUN_DIR}/train_output"
  mkdir -p "${RUN_DIR}"

  if [ -e "${TRAIN_OUT}" ]; then
    echo "[SKIP] bs=${BS}: ${TRAIN_OUT} already exists"
    continue
  fi

  echo "[START] bs=${BS} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw --format=csv,nounits -l 1 > "${RUN_DIR}/gpu.csv" &
  MON_PID=$!

  START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  START_S="$(date +%s)"

  set +e
  HF_HOME=/root/autodl-tmp/huggingface \
  HF_LEROBOT_HOME=/root/autodl-tmp/huggingface/lerobot \
  HF_DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  HF_HUB_DISABLE_XET=1 \
  WANDB_MODE=disabled \
  PYTHONUNBUFFERED=1 \
  lerobot-train \
    --config_path="${CONFIG_PATH}" \
    --policy.repo_id="smolvla-libero-bs-ablation-config-bs${BS}" \
    --policy.pretrained_path=lerobot/smolvla_base \
    --dataset.root="${SNAP}" \
    --dataset.streaming=false \
    --dataset.video_backend=torchcodec \
    --batch_size="${BS}" \
    --num_workers=8 \
    --prefetch_factor=4 \
    --persistent_workers=true \
    --steps="${TOTAL_STEPS}" \
    --log_freq="${LOG_FREQ}" \
    --save_checkpoint=false \
    --save_freq="${TOTAL_STEPS}" \
    --eval_freq=-1 \
    --wandb.enable=false \
    --output_dir="${TRAIN_OUT}" \
    > "${RUN_DIR}/train.log" 2>&1
  RC=$?
  set -e

  END_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  END_S="$(date +%s)"
  kill "${MON_PID}" >/dev/null 2>&1 || true
  wait "${MON_PID}" >/dev/null 2>&1 || true

  if [ "${RC}" -eq 0 ]; then
    STATUS="completed"
  else
    STATUS="failed"
  fi

  cat > "${RUN_DIR}/run_meta.json" <<EOF
{
  "batch_size": ${BS},
  "status": "${STATUS}",
  "return_code": ${RC},
  "start_utc": "${START_ISO}",
  "end_utc": "${END_ISO}",
  "wall_seconds": $((END_S - START_S)),
  "total_steps": ${TOTAL_STEPS},
  "log_freq": ${LOG_FREQ}
}
EOF

  python "${EXP_DIR}/parse_results.py" "${EXP_DIR}" || true
  echo "[DONE] bs=${BS} status=${STATUS} rc=${RC} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

python "${EXP_DIR}/parse_results.py" "${EXP_DIR}"
echo "[ALL_DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
