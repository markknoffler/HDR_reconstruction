#!/usr/bin/env bash
# Fully autonomous Stage-2 training loop: monitor → evaluate → fix → retrain until FHDR targets.
set -euo pipefail

PROJECT="/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/TriGate-HDR"
GUIDE="$PROJECT/TRIGATE_PIPELINE_GUIDE.md"
LDR_DIR="/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/LDR_in"
HDR_DIR="/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/HDR_gt"
V3_BEST="$PROJECT/experiments/stage2_cold_lorcd_v3/best_epoch_30.pt"
INTERVAL=300
TARGET_PSNR=17
TARGET_SSIM=0.71
MAX_ITERATIONS=5
AUTO_LOG="$PROJECT/experiments/stage2_autonomous.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$AUTO_LOG"; }

source "$(conda info --base)/etc/profile.d/conda.sh"

find_train_pid() {
  local ckpt_dir="$1"
  pgrep -f "train_stage2_crf_recovery.*${ckpt_dir}" | head -1 || true
}

get_csv_state() {
  local csv="$1"
  python3 - "$csv" <<'PY'
import csv, sys
csv_path = sys.argv[1]
rows = []
with open(csv_path) as f:
    for row in csv.DictReader(f):
        rows.append(row)
if not rows:
    print("0,0,0,0")
    sys.exit(0)
last = rows[-1]
full = [r for r in rows if r.get("full_val_ran") == "1"]
fv = full[-1] if full else {"epoch": "0", "val_psnr": "0", "val_ssim": "0"}
print(f"{last['epoch']},{fv['epoch']},{fv['val_psnr']},{fv['val_ssim']}")
PY
}

check_log_errors() {
  tail -n 800 "$1" 2>/dev/null | grep -iE "Traceback|OutOfMemory|CUDA out of memory|Killed" | tail -3 || true
}

start_training() {
  local ckpt_dir="$1"
  local warm_from="$2"
  shift 2
  mkdir -p "$ckpt_dir"
  if [[ -f "$PROJECT/experiments/stage2_cold_lorcd_v3/split_manifest.json" && ! -f "$ckpt_dir/split_manifest.json" ]]; then
    cp "$PROJECT/experiments/stage2_cold_lorcd_v3/split_manifest.json" "$ckpt_dir/"
  fi
  log "START $ckpt_dir warm=$warm_from extra=$*"
  cd "$PROJECT"
  export PYTHONPATH="$PROJECT"
  export CUDA_VISIBLE_DEVICES=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  nohup conda run -n deeplearning --no-capture-output python -u -m model.training_scripts.train_stage2_crf_recovery \
    --ldr_dir "$LDR_DIR" \
    --hdr_dir "$HDR_DIR" \
    --checkpoint_dir "$ckpt_dir" \
    --warm_start_from "$warm_from" \
    --epochs 80 --batch_size 1 --max_dim 512 --num_workers 4 \
    --vae_warmup_epochs 0 --full_val_every 5 --train_eval_samples 50 \
    --inference_timesteps 50 --amp --early_stop_patience 3 \
    "$@" >> "$ckpt_dir/train.log" 2>&1 &
  sleep 15
}

wait_for_milestone() {
  local csv="$1" want_epoch="$2" timeout_iters=120
  local i=0
  while (( i < timeout_iters )); do
    if [[ ! -f "$csv" ]]; then sleep "$INTERVAL"; ((i++)); continue; fi
    local state
    state=$(get_csv_state "$csv")
    IFS=, read -r csv_epoch fv_epoch fv_psnr fv_ssim <<< "$state"
    if [[ "$fv_epoch" -ge "$want_epoch" ]]; then
      echo "$fv_epoch,$fv_psnr,$fv_ssim"
      return 0
    fi
    sleep "$INTERVAL"
    ((i++))
  done
  get_csv_state "$csv" | awk -F, '{print $2","$3","$4}'
}

update_guide() {
  local note="$1"
  python3 - "$GUIDE" "$note" <<'PY'
import sys
from pathlib import Path
from datetime import datetime
guide = Path(sys.argv[1])
note = sys.argv[2]
text = guide.read_text()
stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
marker = "### Autonomous loop log"
line = f"- **{stamp}**: {note}\n"
if marker not in text:
    insert = text.find("### Verification status")
    block = f"\n{marker}\n\n{line}"
    text = text[:insert] + block + text[insert:] if insert >= 0 else text + block
else:
    text = text.replace(marker + "\n\n", marker + "\n\n" + line, 1)
guide.write_text(text)
PY
}

run_experiment() {
  local name="$1"
  local ckpt_dir="$PROJECT/experiments/$name"
  local warm_from="$2"
  shift 2
  local extra_args=("$@")

  log "=== Experiment $name ==="
  update_guide "Started **$name** (warm_start=$warm_from, args=${extra_args[*]:-default})"

  if [[ -z "$(find_train_pid "$ckpt_dir")" ]]; then
    start_training "$ckpt_dir" "$warm_from" "${extra_args[@]}"
  else
    log "Training already running for $name"
  fi

  # Wait for epoch 5 full val
  log "Waiting for epoch 5 full val..."
  local m5
  m5=$(wait_for_milestone "$ckpt_dir/training_metrics.csv" 5)
  IFS=, read -r e5 psnr5 ssim5 <<< "$m5"
  log "Epoch 5: PSNR=$psnr5 SSIM=$ssim5"
  update_guide "**$name** epoch 5 full-val: PSNR=$psnr5 SSIM=$ssim5"

  if python3 -c "p=float('$psnr5'); s=float('$ssim5'); exit(0 if p>=$TARGET_PSNR and s>=$TARGET_SSIM else 1)"; then
    log "TARGET at epoch 5"
    echo "$ckpt_dir/best.pt"
    return 0
  fi

  # Wait for epoch 10 or early stop
  log "Waiting for epoch 10 or training end..."
  local i=0
  while (( i < 200 )); do
    local pid
    pid=$(find_train_pid "$ckpt_dir")
    local state
    state=$(get_csv_state "$ckpt_dir/training_metrics.csv")
    IFS=, read -r csv_epoch fv_epoch fv_psnr fv_ssim <<< "$state"
    local err
    err=$(check_log_errors "$ckpt_dir/train.log")

    if [[ -n "$err" && -z "$pid" ]]; then
      log "Crashed: $err — restarting from best"
      local ws="$ckpt_dir/best.pt"
      [[ -f "$ws" ]] || ws="$warm_from"
      start_training "$ckpt_dir" "$ws" "${extra_args[@]}"
    fi

    if [[ "$fv_epoch" -ge 10 ]]; then
      log "Epoch 10: PSNR=$fv_psnr SSIM=$fv_ssim"
      update_guide "**$name** epoch 10 full-val: PSNR=$fv_psnr SSIM=$fv_ssim"
      if python3 -c "p=float('$fv_psnr'); s=float('$fv_ssim'); exit(0 if p>=$TARGET_PSNR and s>=$TARGET_SSIM else 1)"; then
        echo "$ckpt_dir/best.pt"
        return 0
      fi
      break
    fi

    if [[ -z "$pid" ]] && tail -n 50 "$ckpt_dir/train.log" | grep -q "\[early_stop\]"; then
      log "Early stop detected"
      break
    fi
    sleep "$INTERVAL"
    ((i++))
  done

  # Wait for training to finish if still running
  while [[ -n "$(find_train_pid "$ckpt_dir")" ]]; do sleep "$INTERVAL"; done

  local final
  final=$(get_csv_state "$ckpt_dir/training_metrics.csv")
  IFS=, read -r _ fv_epoch fv_psnr fv_ssim <<< "$final"
  log "Finished $name: epoch=$fv_epoch PSNR=$fv_psnr SSIM=$fv_ssim"
  update_guide "**$name** finished: full-val epoch=$fv_epoch PSNR=$fv_psnr SSIM=$fv_ssim"

  if python3 -c "p=float('$fv_psnr'); s=float('$fv_ssim'); exit(0 if p>=$TARGET_PSNR and s>=$TARGET_SSIM else 1)"; then
    echo "$ckpt_dir/best.pt"
    return 0
  fi

  local best="$ckpt_dir/best.pt"
  [[ -f "$best" ]] && echo "$best" || echo "$warm_from"
  return 1
}

log "Autonomous loop started"
update_guide "Autonomous training loop started (target PSNR≥$TARGET_PSNR, SSIM≥$TARGET_SSIM, GPU 0, deeplearning env)"

# v4/v4b/v5 failed (aggressive losses collapsed weights; v5 arch mismatch). v6+ conservative.
log "v4/v4b done — starting v6 conservative fine-tune from v3 best"
best_ws2=$(run_experiment "stage2_cold_lorcd_v6" "$V3_BEST" \
  --cold_lr 5e-5 --mu_psnr_loss_weight 0.5 --hdr_loss_weight 1.0 --exp_loss_weight 2.0 \
  --anchor_exp_weight 0.5 --anchor_hdr_weight 0.5 --radiometric_weight 0.02 \
  --early_stop_patience 5 --inference_timesteps 50) || best_ws2="$V3_BEST"

if [[ -f "${best_ws2:-}" ]] && python3 -c "
import torch
ck=torch.load('$best_ws2', map_location='cpu')
p=float(ck.get('best_val_psnr',0) or 0); s=float(ck.get('best_val_ssim',0) or 0)
exit(0 if p>=$TARGET_PSNR and s>=$TARGET_SSIM else 1)
" 2>/dev/null; then
  log "SUCCESS v6: $best_ws2"
  update_guide "SUCCESS v6 — target reached. best.pt=$best_ws2"
  exit 0
fi

best_ws3="${best_ws2:-$V3_BEST}"
[[ -f "$best_ws3" ]] || best_ws3="$V3_BEST"
log "Starting v7 low-LR polish"
best_ws4=$(run_experiment "stage2_cold_lorcd_v7" "$best_ws3" \
  --cold_lr 2e-5 --mu_psnr_loss_weight 1.0 --radiometric_weight 0 \
  --early_stop_patience 5) || best_ws4="$best_ws3"

log "Autonomous loop completed. Best: ${best_ws4:-unknown}"
update_guide "Loop done. Best: ${best_ws4:-unknown}. v4/v4b collapsed from heavy losses; v6 uses conservative recipe."
