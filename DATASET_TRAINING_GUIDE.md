# TriGate-HDR Dataset Training Guide

This file summarizes:
- which datasets are downloaded and ready,
- where training outputs are saved according to code,
- exact training commands for each dataset.

## Current dataset status

- Downloaded and ready:
  - `datasets/HDR-REAL` (1838 pairs)
  - `datasets/HDR-EYE` (46 pairs)
  - `datasets/HDR-REAL-FullTrain` (9795 pairs; symlink to `SingleHDR_training_data/HDR-Real`)
- Not downloaded / not ready:
  - `datasets/HDR-Synth` (0 pairs)
  - `datasets/AIM2025` (0 pairs)

## Where results are saved (from `train_stage2_crf_recovery.py`)

For each run, outputs are saved under the command's `--checkpoint_dir`:

- Checkpoints:
  - `best.pt`
  - `latest.pt`
  - `epoch_*.pt`
  - `best_epoch_*.pt`
- Metrics CSV:
  - `training_metrics.csv`
- Expo benchmark CSV (when `--benchmark_metrics` is used):
  - file from `--benchmark_metrics_csv` (inside `checkpoint_dir`)
- Validation image exports:
  - `validation_results/`
- Final test exports:
  - `final_test_exports/`

## Training commands

Run from:
`/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/TriGate-HDR`

Activate env first:
`conda activate deeplearning`

### 1) HDR-REAL benchmark (downloaded)

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m model.training_scripts.train_stage2_crf_recovery \
    --ldr_dir "$HOME/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/datasets/HDR-REAL/LDR_in" \
    --hdr_dir "$HOME/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/datasets/HDR-REAL/HDR_gt" \
    --checkpoint_dir "$HOME/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/TriGate-HDR/experiments/trigate_v2_gpure_hdr_real" \
    --arch_v2 --use_rso --use_lr_cfp --amp \
    --batch_size 1 --max_dim 512 --num_workers 2 \
    --cold_lr 1e-5 --ema_decay 0 --vae_warmup_epochs 0 --freeze_vae_after_warmup \
    --val_ratio 0.2 --split_seed 42 --skip_trial_validation \
    --full_val_every 5 --train_eval_samples 50 --val_eval_samples 50 \
    --early_stop_patience 8 --inference_timesteps 25 \
    --inference_loss_weight 0 \
    --benchmark_metrics \
    --benchmark_metrics_csv "trigate_v2_gpure_hdr_real_benchmark_metrics.csv" \
    --epochs 1359
```

### 2) HDR-EYE benchmark (downloaded)

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m model.training_scripts.train_stage2_crf_recovery \
    --ldr_dir "$HOME/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/datasets/HDR-EYE/LDR_in" \
    --hdr_dir "$HOME/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/datasets/HDR-EYE/HDR_gt" \
    --checkpoint_dir "$HOME/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/TriGate-HDR/experiments/trigate_v2_gpure_hdr_eye" \
    --arch_v2 --use_rso --use_lr_cfp --amp \
    --batch_size 1 --max_dim 512 --num_workers 2 \
    --cold_lr 1e-5 --ema_decay 0 --vae_warmup_epochs 0 --freeze_vae_after_warmup \
    --val_ratio 0.2 --split_seed 42 --skip_trial_validation \
    --full_val_every 5 --train_eval_samples 50 --val_eval_samples 50 \
    --early_stop_patience 8 --inference_timesteps 25 \
    --inference_loss_weight 0 \
    --benchmark_metrics \
    --benchmark_metrics_csv "trigate_v2_gpure_hdr_eye_benchmark_metrics.csv" \
    --epochs 300
```

### 3) HDR-REAL-FullTrain (downloaded; full 9795-pair set)

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m model.training_scripts.train_stage2_crf_recovery \
    --ldr_dir "$HOME/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/LDR_in" \
    --hdr_dir "$HOME/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/HDR_gt" \
    --checkpoint_dir "$HOME/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/TriGate-HDR/experiments/trigate_v2_gpure_hdr_real_full" \
    --arch_v2 --use_rso --use_lr_cfp --amp \
    --batch_size 1 --max_dim 512 --num_workers 2 \
    --cold_lr 1e-5 --ema_decay 0 --vae_warmup_epochs 0 --freeze_vae_after_warmup \
    --val_ratio 0.2 --split_seed 42 --skip_trial_validation \
    --full_val_every 5 --train_eval_samples 50 --val_eval_samples 50 \
    --early_stop_patience 8 --inference_timesteps 25 \
    --inference_loss_weight 0 \
    --benchmark_metrics \
    --benchmark_metrics_csv "trigate_v2_gpure_hdr_real_full_benchmark_metrics.csv" \
    --epochs 256
```

## Not downloaded datasets and how to prepare

### HDR-Synth (not ready)

- Expected location: `datasets/HDR-Synth/LDR_in` and `datasets/HDR-Synth/HDR_gt`
- Status: not downloaded (0 pairs)
- Prepare:
  1. Download `SingleHDR-Training.zip` manually.
  2. Place it at `datasets/_downloads/SingleHDR-Training.zip`
  3. Run:

```bash
cd TriGate-HDR
python scripts/download_expo_datasets.py --dataset hdr_synth --prepare-only
```

### AIM2025 (not ready)

- Expected location: `datasets/AIM2025/LDR_in` and `datasets/AIM2025/HDR_gt`
- Status: not downloaded (0 pairs)
- Prepare using your Codabench training zip:

```bash
cd TriGate-HDR
python scripts/download_expo_datasets.py --dataset aim2025 --aim2025-url /absolute/path/to/aim2025_train.zip
```
