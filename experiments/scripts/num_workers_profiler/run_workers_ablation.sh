#!/usr/bin/env bash
set -u
EXP_DIR="/root/autodl-tmp/lerobot/outputs/libero_workers_gpu_profiler_20260627"
CONFIG_PATH="/root/autodl-tmp/lerobot/outputs/smolvla_libero_base_direct_5epoch_bs24_56975steps_explicit_20260520_2225utc/checkpoints/045580/pretrained_model/train_config.json"
SNAP="/root/autodl-tmp/huggingface/lerobot/hub/datasets--HuggingFaceVLA--libero/snapshots/86958911c0f959db2bbbdb107eb3e17c5f9c798e"
WORKERS=(0 2 4 8 16)
BS=24
TOTAL_STEPS=120
LOG_FREQ=10
mkdir -p "${EXP_DIR}"
cat > "${EXP_DIR}/experiment_config.json" <<EOF
{
  "task": "LIBERO SmolVLA num_workers ablation + GPU utilization",
  "date_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "config_path": "${CONFIG_PATH}",
  "batch_size": ${BS},
  "num_workers": [0, 2, 4, 8, 16],
  "total_steps": ${TOTAL_STEPS},
  "warmup_steps": 20,
  "measurement_steps": 100,
  "dataset_num_frames": 273465,
  "notes": "Same LIBERO SmolVLA model/data/batch/optimizer config as batch-size experiment; only num_workers changes. Short-window timing benchmark because num_workers=0 makes the 1500-step window take several hours. Epoch time is estimated from stable mean step time."
}
EOF
for NW in "${WORKERS[@]}"; do
  RUN_DIR="${EXP_DIR}/nw${NW}"
  TRAIN_OUT="${RUN_DIR}/train_output"
  mkdir -p "${RUN_DIR}"
  if [ -e "${RUN_DIR}/run_meta.json" ]; then
    echo "[SKIP] nw=${NW}: already has run_meta.json"
    continue
  fi
  echo "[START] nw=${NW} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw --format=csv,nounits -l 1 > "${RUN_DIR}/gpu.csv" &
  MON_PID=$!
  START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; START_S="$(date +%s)"
  set +e
  HF_HOME=/root/autodl-tmp/huggingface \
  HF_LEROBOT_HOME=/root/autodl-tmp/huggingface/lerobot \
  HF_DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 \
  WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
  lerobot-train \
    --config_path="${CONFIG_PATH}" \
    --policy.repo_id="smolvla-libero-workers-nw${NW}" \
    --policy.pretrained_path=lerobot/smolvla_base \
    --dataset.root="${SNAP}" \
    --dataset.streaming=false \
    --dataset.video_backend=torchcodec \
    --batch_size="${BS}" \
    --num_workers="${NW}" \
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
  END_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; END_S="$(date +%s)"
  kill "${MON_PID}" >/dev/null 2>&1 || true
  wait "${MON_PID}" >/dev/null 2>&1 || true
  if [ "${RC}" -eq 0 ]; then STATUS="completed"; else STATUS="failed"; fi
  cat > "${RUN_DIR}/run_meta.json" <<EOF
{"num_workers": ${NW}, "status": "${STATUS}", "return_code": ${RC}, "start_utc": "${START_ISO}", "end_utc": "${END_ISO}", "wall_seconds": $((END_S - START_S)), "batch_size": ${BS}, "total_steps": ${TOTAL_STEPS}, "warmup_steps": 20, "measurement_steps": 100, "log_freq": ${LOG_FREQ}}
EOF
  python "${EXP_DIR}/parse_workers_results.py" "${EXP_DIR}" || true
  echo "[DONE] nw=${NW} status=${STATUS} rc=${RC} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done
python "${EXP_DIR}/parse_workers_results.py" "${EXP_DIR}"
echo "[ALL_DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
