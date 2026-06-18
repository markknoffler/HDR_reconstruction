#!/usr/bin/env bash
# Watch v8; restart from best.pt if process dies.
set -euo pipefail
PROJECT="/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/TriGate-HDR"
CKPT="$PROJECT/experiments/stage2_cold_lorcd_v8"
V3E30="$PROJECT/experiments/stage2_cold_lorcd_v3/epoch_30.pt"
LOG="$PROJECT/experiments/stage2_autonomous.log"
LDR="/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/LDR_in"
HDR="/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/HDR_gt"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

start() {
  local resume="$1"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  cd "$PROJECT"
  export PYTHONPATH="$PROJECT" CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  nohup conda run -n deeplearning --no-capture-output python -u -m model.training_scripts.train_stage2_crf_recovery \
    --ldr_dir "$LDR" --hdr_dir "$HDR" --checkpoint_dir "$CKPT" \
    --continue_train --resume_from "$resume" \
    --epochs 90 --batch_size 1 --max_dim 512 --num_workers 4 \
    --vae_warmup_epochs 8 --full_val_every 5 --train_eval_samples 50 \
    --inference_timesteps 50 --amp --early_stop_patience 5 --cold_lr 2e-5 \
    --inference_loss_weight 1.0 --train_inference_steps 10 --inference_loss_every 100 --ema_decay 0 \
    >> "$CKPT/train.log" 2>&1 &
  sleep 10
}

log "watch_v8 started"
while true; do
  pid=$(pgrep -f "train_stage2_crf_recovery.*stage2_cold_lorcd_v8" | head -1 || true)
  if [[ -f "$CKPT/training_metrics.csv" ]]; then
    tail -1 "$CKPT/training_metrics.csv" | tee -a "$LOG"
  fi
  if [[ -z "$pid" ]]; then
    if tail -50 "$CKPT/train.log" 2>/dev/null | grep -q "\[early_stop\]"; then
      log "v8 early-stopped — done"
      break
    fi
    resume="$CKPT/best.pt"
    [[ -f "$resume" ]] || resume="$V3E30"
    log "v8 dead — restarting from $resume"
    start "$resume"
  fi
  if [[ -f "$CKPT/training_metrics.csv" ]]; then
  python3 -c "
import csv
rows=list(csv.DictReader(open('$CKPT/training_metrics.csv')))
full=[r for r in rows if r.get('full_val_ran')=='1']
if full:
  r=full[-1]
  p,s=float(r['val_psnr']),float(r['val_ssim'])
  if p>=17 and s>=0.71:
    open('$CKPT/SUCCESS','w').write(f'{p},{s}')
" && [[ -f "$CKPT/SUCCESS" ]] && log "TARGET REACHED $(cat $CKPT/SUCCESS)" && break
  fi
  sleep 300
done
