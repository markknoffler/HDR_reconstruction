#!/usr/bin/env bash
# Autonomous Stage 2 v4 training monitor — checks every 5 minutes.
set -euo pipefail

PROJECT="/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/TriGate-HDR"
CKPT_DIR="$PROJECT/experiments/stage2_cold_lorcd_v4"
LOG="$CKPT_DIR/train.log"
CSV="$CKPT_DIR/training_metrics.csv"
MONITOR_LOG="$CKPT_DIR/monitor.log"
GUIDE="$PROJECT/TRIGATE_PIPELINE_GUIDE.md"
INTERVAL=300  # 5 minutes
TARGET_PSNR=17
TARGET_SSIM=0.71
V4B_TRIGGER_PSNR=12
V4B_EPOCH=20

LDR_DIR="/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/LDR_in"
HDR_DIR="/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/HDR_gt"
V3_BEST="$PROJECT/experiments/stage2_cold_lorcd_v3/best_epoch_30.pt"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MONITOR_LOG"; }

find_train_pid() {
  pgrep -f "train_stage2_crf_recovery.*stage2_cold_lorcd_v4" | head -1 || true
}

get_latest_epoch() {
  if [[ ! -f "$CSV" ]]; then echo 0; return; fi
  tail -1 "$CSV" | cut -d, -f1
}

get_full_val_metrics() {
  python3 - "$CSV" <<'PY'
import csv, sys
csv_path = sys.argv[1]
rows = []
with open(csv_path) as f:
    for row in csv.DictReader(f):
        if row.get("full_val_ran") == "1":
            rows.append(row)
if not rows:
    print("0,0,0")
else:
    r = rows[-1]
    print(f"{r['epoch']},{r['val_psnr']},{r['val_ssim']}")
PY
}

check_log_errors() {
  tail -n 500 "$LOG" 2>/dev/null | grep -iE "Traceback|OutOfMemory|CUDA out of memory|nan|NaN|RuntimeError|Killed" | tail -5 || true
}

warm_start_path() {
  if [[ -f "$CKPT_DIR/best.pt" ]]; then echo "$CKPT_DIR/best.pt"
  elif [[ -f "$V3_BEST" ]]; then echo "$V3_BEST"
  else echo ""; fi
}

start_training() {
  local ckpt_dir="$1"
  local warm_from="$2"
  shift 2
  local extra_args=("$@")
  log "Starting training in $ckpt_dir warm_start=$warm_from"
  cd "$PROJECT"
  export PYTHONPATH="$PROJECT"
  export CUDA_VISIBLE_DEVICES=0
  nohup conda run -n deeplearning --no-capture-output python -u -m model.training_scripts.train_stage2_crf_recovery \
    --ldr_dir "$LDR_DIR" \
    --hdr_dir "$HDR_DIR" \
    --checkpoint_dir "$ckpt_dir" \
    --warm_start_from "$warm_from" \
    --epochs 80 --batch_size 1 --max_dim 512 --num_workers 4 \
    --vae_warmup_epochs 0 --full_val_every 5 --train_eval_samples 50 \
    --inference_timesteps 50 --amp --early_stop_patience 3 \
    "${extra_args[@]}" \
    >> "$ckpt_dir/train.log" 2>&1 &
  sleep 10
}

restart_if_dead() {
  local pid
  pid=$(find_train_pid)
  if [[ -n "$pid" ]]; then return 0; fi
  log "Training process not found — restarting v4"
  local ws
  ws=$(warm_start_path)
  if [[ -z "$ws" ]]; then log "ERROR: no warm-start checkpoint"; return 1; fi
  start_training "$CKPT_DIR" "$ws"
}

update_guide_milestone() {
  local epoch="$1" psnr="$2" ssim="$3" note="$4"
  python3 - <<PY
import re
from pathlib import Path
guide = Path("$GUIDE")
text = guide.read_text()
row = f"| Epoch {epoch} | {psnr} | {ssim} | {note} |"
# Insert into v4 milestones table if not already present
if f"Epoch {epoch} |" not in text or "v4 run milestones" in text:
    marker = "### v4 run milestones"
    if marker not in text:
        insert_at = text.find("### Verification status")
        block = f"\n{marker} (auto-updated)\n\n| Milestone | Full-val PSNR | Full-val SSIM | Notes |\n|-----------|---------------|---------------|-------|\n{row}\n\n"
        if insert_at >= 0:
            text = text[:insert_at] + block + text[insert_at:]
        else:
            text += block
    else:
        if f"| Epoch {epoch} |" not in text:
            text = text.replace(
                "|-----------|---------------|---------------|-------|",
                f"|-----------|---------------|---------------|-------|\n{row}",
                1,
            )
    guide.write_text(text)
PY
}

training_done() {
  tail -n 100 "$LOG" 2>/dev/null | grep -qE "\[early_stop\]|Training complete|Finished epoch 80|Saved best\.pt" && return 0
  local epoch
  epoch=$(get_latest_epoch)
  [[ "$epoch" -ge 80 ]] && return 0
  return 1
}

v4b_started=false
last_milestone_epoch=-1

log "Monitor started. PID=$(find_train_pid) interval=${INTERVAL}s"

while true; do
  pid=$(find_train_pid)
  epoch=$(get_latest_epoch)
  metrics=$(get_full_val_metrics)
  IFS=, read -r fv_epoch fv_psnr fv_ssim <<< "$metrics"
  errors=$(check_log_errors)

  log "CHECK pid=${pid:-DEAD} csv_epoch=$epoch full_val_epoch=$fv_epoch psnr=$fv_psnr ssim=$fv_ssim"

  if [[ -n "$errors" ]]; then
    log "LOG ERRORS (recent):"
    echo "$errors" | tee -a "$MONITOR_LOG"
  fi

  if [[ -z "$pid" ]]; then
    if training_done; then
      log "Training finished (no process, done signal in log)"
      break
    fi
    restart_if_dead || true
    pid=$(find_train_pid)
  fi

  # Milestone updates at full-val epochs (5,10,15,...)
  if [[ "$fv_epoch" != "0" && "$fv_epoch" != "$last_milestone_epoch" ]]; then
    update_guide_milestone "$fv_epoch" "$fv_psnr" "$fv_ssim" "full validation"
    last_milestone_epoch="$fv_epoch"
    log "Updated guide milestone epoch=$fv_epoch"
  fi

  # Target reached?
  if python3 -c "import sys; p=float('$fv_psnr'); s=float('$fv_ssim'); sys.exit(0 if p>=$TARGET_PSNR and s>=$TARGET_SSIM else 1)" 2>/dev/null; then
    log "TARGET REACHED psnr=$fv_psnr ssim=$fv_ssim"
    break
  fi

  # v4b fork after epoch 20 if PSNR still low
  if [[ "$v4b_started" == "false" && "$epoch" -ge "$V4B_EPOCH" ]]; then
    if python3 -c "import sys; sys.exit(0 if float('$fv_psnr') < $V4B_TRIGGER_PSNR else 1)" 2>/dev/null; then
      log "Epoch>=20 and full-val PSNR=$fv_psnr < $V4B_TRIGGER_PSNR — v4b fork needed (handled by parent agent)"
      echo "V4B_NEEDED" >> "$MONITOR_LOG"
      v4b_started=true
    fi
  fi

  if training_done; then
    log "Training complete signal detected"
    break
  fi

  sleep "$INTERVAL"
done

# Final summary
best_pt="$CKPT_DIR/best.pt"
if [[ -f "$best_pt" ]]; then
  log "FINAL best.pt=$best_pt metrics=$metrics"
else
  log "FINAL no best.pt yet metrics=$metrics"
fi
echo "MONITOR_DONE" >> "$MONITOR_LOG"
