"""
Train Stage-2 ColdEfficient-LORCD (latent cold diffusion, no foundation model).

Validation metrics: FHDR/test.py PSNR-μ and SSIM via common_training.compute_psnr_ssim.
"""

import argparse
import csv
import os
import time

# Must be set before the first CUDA allocation (reduces fragmentation after OOM).
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..decoders.cold_hdr_diffusion_decoder import ColdHDRDiffusion
from ..losses.stage_composite_losses import stage2_loss
from ..losses.radiometric_losses import HybridRadiometricConsistencyLoss
from .common_training import (
    HDRVDPMetrics,
    ModelEMA,
    add_subset_args,
    apply_smoke_test_args,
    default_hrishav_data_paths,
    maybe_resume,
    mse_loss,
    mu_tonemap,
    reset_cuda_memory,
    print_epoch_summary,
    save_best_checkpoint,
    save_checkpoint,
    save_latest_checkpoint,
    sanitize_data_path,
)
from .dataset_splits import build_dataloaders
from .val_export import export_final_test_samples, make_stage2_epoch_predictor, validate_model_mtraining


def _make_subset_loader(loader, sample_count: int, num_workers: int, seed: int):
    ds = loader.dataset
    if hasattr(ds, "indices"):
        indices = list(ds.indices)
        base_ds = ds.dataset
    else:
        indices = list(range(len(ds)))
        base_ds = ds
    if not indices:
        return None
    if len(indices) <= sample_count:
        picked = indices
    else:
        g = torch.Generator()
        g.manual_seed(int(seed))
        order = torch.randperm(len(indices), generator=g).tolist()
        picked = [indices[i] for i in order[:sample_count]]
    subset = Subset(base_ds, picked)
    return DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def _append_stage2_metrics_csv(
    csv_path,
    epoch,
    train_loss,
    tr_psnr,
    tr_ssim,
    tr_h2,
    tr_h3,
    val_psnr,
    val_ssim,
    val_h2,
    val_h3,
    val_ran: bool,
    vae_only: bool,
    metric_note: str = "",
):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        fieldnames = [
            "epoch",
            "train_loss",
            "vae_only",
            "train_probe_psnr",
            "train_probe_ssim",
            "train_probe_hdrvdp2",
            "train_probe_hdrvdp3",
            "val_psnr",
            "val_ssim",
            "val_hdrvdp2",
            "val_hdrvdp3",
            "full_val_ran",
            "metric_note",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": f"{float(train_loss):.6f}",
                "vae_only": int(vae_only),
                "train_probe_psnr": f"{float(tr_psnr):.4f}",
                "train_probe_ssim": f"{float(tr_ssim):.4f}",
                "train_probe_hdrvdp2": f"{float(tr_h2):.4f}",
                "train_probe_hdrvdp3": f"{float(tr_h3):.4f}",
                "val_psnr": f"{float(val_psnr):.4f}" if val_ran else "",
                "val_ssim": f"{float(val_ssim):.4f}" if val_ran else "",
                "val_hdrvdp2": f"{float(val_h2):.4f}" if val_ran else "",
                "val_hdrvdp3": f"{float(val_h3):.4f}" if val_ran else "",
                "full_val_ran": int(val_ran),
                "metric_note": metric_note,
            }
        )
    print(f"  Metrics CSV updated: {csv_path}")


def main():
    _defaults = default_hrishav_data_paths()
    parser = argparse.ArgumentParser(
        description="Train Stage-2 ColdEfficient-LORCD (latent expansion cold diffusion).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--cold_lr",
        type=float,
        default=0.0,
        help="LR for cold UNet after VAE freeze (0 = lr * 0.1).",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max grad norm; 0 disables.")
    parser.add_argument("--checkpoint_dir", type=str, default=_defaults["checkpoint_dir_stage2"])
    parser.add_argument("--ldr_dir", type=str, default=_defaults["ldr_dir"])
    parser.add_argument("--hdr_dir", type=str, default=_defaults["hdr_dir"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_dim", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=100, help="Cold diffusion steps (training).")
    parser.add_argument(
        "--inference_timesteps",
        type=int,
        default=50,
        help="Cold reverse steps for restore_hdr at validation (lower saves VRAM/time).",
    )
    parser.add_argument("--base_ch", type=int, default=64, help="Latent UNet base channels.")
    parser.add_argument("--latent_ch", type=int, default=4, help="MiniHDR-VAE latent channels.")
    parser.add_argument("--vae_warmup_epochs", type=int, default=8, help="VAE-only warmup before cold training.")
    parser.add_argument("--hdr_loss_weight", type=float, default=1.0, help="Pixel HDR L1 weight (cold phase).")
    parser.add_argument("--cold_loss_weight", type=float, default=1.0)
    parser.add_argument("--exp_loss_weight", type=float, default=2.0)
    parser.add_argument("--trust_loss_weight", type=float, default=0.01)
    parser.add_argument(
        "--anchor_exp_weight",
        type=float,
        default=0.5,
        help="Extra latent expansion loss at t=T-1 (restore_hdr start state).",
    )
    parser.add_argument(
        "--anchor_hdr_weight",
        type=float,
        default=0.5,
        help="Extra pixel HDR loss at t=T-1 (restore_hdr start state).",
    )
    parser.add_argument(
        "--mu_psnr_loss_weight",
        type=float,
        default=0.5,
        help="MSE on FHDR mu_tonemap(pred) vs mu_tonemap(gt) — aligns training with PSNR-μ metric.",
    )
    parser.add_argument(
        "--ssim_rgb_l1_weight",
        type=float,
        default=0.0,
        help="L1 on (pred+1)/2 vs (gt+1)/2 — FHDR SSIM input space; improves structural fidelity.",
    )
    parser.add_argument(
        "--freeze_vae_after_warmup",
        action="store_true",
        default=True,
        help="Freeze mini VAE/MLN after warmup so cold UNet learns expansion (recommended).",
    )
    parser.add_argument(
        "--no_freeze_vae_after_warmup",
        action="store_false",
        dest="freeze_vae_after_warmup",
        help="Keep VAE trainable during cold diffusion (can hide UNet errors).",
    )
    parser.add_argument("--ms_cold_weight", type=float, default=0.25)
    parser.add_argument("--vae_loss_weight", type=float, default=0.1)
    parser.add_argument("--mono_loss_weight", type=float, default=0.01)
    parser.add_argument("--radiometric_weight", type=float, default=0.05)
    parser.add_argument(
        "--radiometric_decay_epochs",
        type=int,
        default=25,
        help="Linearly decay radiometric_weight to 0 over this many cold epochs.",
    )
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA for validation; 0 disables.")
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=5,
        help="Stop after N full validations without PSNR improvement (0 disables).",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--continue_train", action="store_true")
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Checkpoint file to resume from (e.g. epoch_5.pt). Default: checkpoint_dir/latest.pt",
    )
    parser.add_argument(
        "--warm_start_from",
        type=str,
        default="",
        help="Load model weights only and restart at epoch 1 (fresh optimizer). Use best.pt from a prior run.",
    )
    parser.add_argument(
        "--trial_val_samples",
        type=int,
        default=5,
        help="Val images for trial validation before epoch 1.",
    )
    parser.add_argument("--skip_trial_validation", action="store_true")
    parser.add_argument(
        "--train_eval_samples",
        type=int,
        default=50,
        help="Train images for per-epoch metric probe.",
    )
    parser.add_argument(
        "--full_val_every",
        type=int,
        default=10,
        help="Full validation every N epochs (+ final epoch).",
    )
    add_subset_args(parser)
    args = parser.parse_args()
    args = apply_smoke_test_args(args)

    args.ldr_dir = sanitize_data_path(args.ldr_dir)
    args.hdr_dir = sanitize_data_path(args.hdr_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        reset_cuda_memory(device, "startup")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    validation_root = args.validation_results_dir or os.path.join(args.checkpoint_dir, "validation_results")
    os.makedirs(validation_root, exist_ok=True)
    csv_path = os.path.join(args.checkpoint_dir, "training_metrics.csv")

    print(
        f"[Stage2-LORCD] timesteps={args.timesteps} base_ch={args.base_ch} latent_ch={args.latent_ch} "
        f"vae_warmup={args.vae_warmup_epochs} (train from scratch)"
    )
    model = ColdHDRDiffusion(
        timesteps=args.timesteps,
        base_ch=args.base_ch,
        latent_ch=args.latent_ch,
    ).to(device)
    model.inference_timesteps = int(args.inference_timesteps)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    radiometric_loss_fn = HybridRadiometricConsistencyLoss()
    hdrvdp_calculator = HDRVDPMetrics(use_real_hdrvdp=False)

    start_epoch = 1
    best_psnr, best_ssim = 0.0, 0.0
    best_hdrvdp2, best_hdrvdp3 = 0.0, 0.0
    if args.warm_start_from:
        warm_path = sanitize_data_path(args.warm_start_from)
        ckpt = torch.load(warm_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        if ckpt.get("inference_timesteps") is not None:
            model.inference_timesteps = int(ckpt["inference_timesteps"])
        prior_psnr = float(ckpt.get("best_val_psnr", ckpt.get("val_psnr", 0.0)) or 0.0)
        prior_ssim = float(ckpt.get("best_val_ssim", ckpt.get("val_ssim", 0.0)) or 0.0)
        # New experiment dir tracks its own best.pt from scratch (weights only).
        best_psnr, best_ssim = 0.0, 0.0
        print(
            f"[warm_start] Loaded weights from {warm_path} (prior run best PSNR={prior_psnr:.4f}). "
            f"Starting fresh at epoch 1; best tracking reset for this checkpoint_dir."
        )
    elif args.continue_train:
        resume_from = sanitize_data_path(args.resume_from) if args.resume_from else ""
        start_epoch, best_psnr, best_ssim, best_hdrvdp2, best_hdrvdp3 = maybe_resume(
            args.checkpoint_dir, model, optimizer, resume_from=resume_from, strict=False
        )
        model = model.to(device)
        resume_path = resume_from or os.path.join(args.checkpoint_dir, "latest.pt")
        if os.path.isfile(resume_path):
            ckpt_meta = torch.load(resume_path, map_location="cpu")
            if ckpt_meta.get("inference_timesteps") is not None:
                model.inference_timesteps = int(ckpt_meta["inference_timesteps"])
        print(
            f"Resuming from epoch {start_epoch} (best PSNR={best_psnr:.4f}), "
            f"inference_timesteps={model.inference_timesteps}"
        )
        if device.type == "cuda":
            reset_cuda_memory(device, "after resume")

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    scheduler = None
    full_val_streak = 0

    if args.vae_warmup_epochs == 0 and args.freeze_vae_after_warmup:
        model.set_vae_trainable(False)
        cold_lr = args.cold_lr if args.cold_lr > 0 else args.lr * 0.1
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cold_lr,
        )
        cold_epochs = max(1, args.epochs)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cold_epochs, eta_min=cold_lr * 0.05
        )
        print(f"[Stage2-LORCD] VAE/MLN frozen from start (lr={cold_lr:.2e}, cosine schedule).")

    train_loader, val_loader, full_dataset, val_idx = build_dataloaders(
        args.ldr_dir,
        args.hdr_dir,
        args.batch_size,
        max_dim=args.max_dim,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        subset_fraction=args.subset_fraction,
        subset_packet=args.subset_packet,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    if args.batch_size > 2:
        print(
            f"[WARN] batch_size={args.batch_size} > 2 — HDR at max_dim={args.max_dim} is unstable "
            f"with large batches; use 1–2 for best PSNR/SSIM."
        )
    print(
        f"[Stage2] train={len(train_loader.dataset)} val={len(val_loader.dataset) if val_loader else 0} "
        f"smoke_test={args.smoke_test}"
    )
    print(f"  Metrics (FHDR/test.py PSNR-μ + SSIM): {csv_path}")
    print(f"  Trial val before epoch 1: {not args.skip_trial_validation and start_epoch == 1}")

    predict_fn = make_stage2_epoch_predictor(model, vae_warmup_epochs=args.vae_warmup_epochs)

    if start_epoch == 1 and not args.skip_trial_validation and val_loader is not None:
        trial_loader = _make_subset_loader(
            val_loader, max(1, args.trial_val_samples), args.num_workers, args.val_export_seed
        )
        if trial_loader is not None:
            print("\n" + "=" * 60)
            print(f"Trial validation before epoch 1 ({len(trial_loader.dataset)} images)")
            print("=" * 60)
            model.eval()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            t_psnr, t_ssim, t_h2, t_h3 = validate_model_mtraining(
                trial_loader,
                device,
                epoch=0,
                hdrvdp_calculator=hdrvdp_calculator,
                predict_hdr=predict_fn,
                validation_root=validation_root,
                save_samples=True,
                max_samples=min(3, len(trial_loader.dataset)),
                amp=args.amp,
            )
            print(f"  Trial PSNR/SSIM/H2/H3: {t_psnr:.4f} / {t_ssim:.4f} / {t_h2:.4f} / {t_h3:.4f}")
            print(f"  Exports: {os.path.join(validation_root, 'epoch_0')}")
            _append_stage2_metrics_csv(
                csv_path,
                epoch=0,
                train_loss=0.0,
                tr_psnr=0.0,
                tr_ssim=0.0,
                tr_h2=0.0,
                tr_h3=0.0,
                val_psnr=t_psnr,
                val_ssim=t_ssim,
                val_h2=t_h2,
                val_h3=t_h3,
                val_ran=True,
                vae_only=True,
                metric_note="trial_val_before_epoch_1",
            )
            print("=" * 60 + "\n")
            model.train()

    for epoch in range(start_epoch, args.epochs + 1):
        predict_fn.set_epoch(epoch)
        epoch_start = time.time()
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Stage2 Cold {epoch}/{args.epochs}", leave=True)
        vae_only = epoch <= args.vae_warmup_epochs
        if epoch == args.vae_warmup_epochs + 1 and args.freeze_vae_after_warmup:
            model.set_vae_trainable(False)
            cold_lr = args.cold_lr if args.cold_lr > 0 else args.lr * 0.1
            optimizer = optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=cold_lr,
            )
            cold_epochs = max(1, args.epochs - args.vae_warmup_epochs)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cold_epochs, eta_min=cold_lr * 0.05
            )
            if ema is not None:
                ema = ModelEMA(model, decay=args.ema_decay)
            print(
                f"[Stage2-LORCD] VAE/MLN frozen — optimizer rebuilt for cold UNet only (lr={cold_lr:.2e})."
            )
        if epoch == args.vae_warmup_epochs + 1 and device.type == "cuda":
            reset_cuda_memory(device, "pre full cold")
            print("[Stage2-LORCD] VAE warmup done — cleared CUDA cache before full cold training.")
        elif device.type == "cuda":
            reset_cuda_memory(device, f"epoch {epoch} start")
        for step, batch in enumerate(pbar, start=1):
            ldr = batch["ldr_image"].to(device)
            hdr = batch["hdr_image"].to(device)
            gate = batch["gate"].to(device)
            optimizer.zero_grad(set_to_none=True)
            use_amp = args.amp and device.type == "cuda" and vae_only
            with autocast("cuda", enabled=use_amp):
                pred, cold_parts = model(hdr, ldr, gate=gate, vae_only=vae_only)
                if vae_only:
                    loss = args.vae_loss_weight * cold_parts["vae_loss"]
                else:
                    vae_w = 0.0 if getattr(model, "vae_frozen", False) else args.vae_loss_weight
                    cold_epoch = max(1, epoch - args.vae_warmup_epochs)
                    rad_w = args.radiometric_weight
                    if args.radiometric_decay_epochs > 0:
                        rad_w *= max(0.0, 1.0 - (cold_epoch - 1) / float(args.radiometric_decay_epochs))
                    loss = (
                        args.hdr_loss_weight * cold_parts["hdr_loss"]
                        + args.cold_loss_weight * cold_parts["cold_loss"]
                        + args.exp_loss_weight * cold_parts["exp_loss"]
                        + args.trust_loss_weight * cold_parts["trust_loss"]
                        + args.ms_cold_weight * cold_parts["ms_cold_loss"]
                        + args.mono_loss_weight * cold_parts["mono_loss"]
                        + args.anchor_exp_weight * cold_parts["anchor_exp"]
                        + args.anchor_hdr_weight * cold_parts["anchor_hdr"]
                        + vae_w * cold_parts["vae_loss"]
                    )
                    if args.mu_psnr_loss_weight > 0:
                        mu_loss = mse_loss(mu_tonemap(pred), mu_tonemap(hdr))
                        loss = loss + args.mu_psnr_loss_weight * mu_loss
                    pred_lin = torch.clamp((pred + 1.0) * 0.5, 0.0, 1.0)
                    if args.ssim_rgb_l1_weight > 0:
                        hdr_lin_ssim = torch.clamp((hdr + 1.0) * 0.5, 0.0, 1.0)
                        loss = loss + args.ssim_rgb_l1_weight * torch.mean(torch.abs(pred_lin - hdr_lin_ssim))
                    hdr_lin = torch.clamp((hdr + 1.0) * 0.5, 0.0, 1.0)
                    rad_loss, _ = stage2_loss(
                        pred_lin, hdr_lin, ldr, gate, radiometric_loss_fn=radiometric_loss_fn
                    )
                    loss = loss + rad_w * rad_loss
            loss_val = float(loss.detach().item())
            if not torch.isfinite(loss):
                print(f"[WARN] skip batch step={step} epoch={epoch}: non-finite loss")
                optimizer.zero_grad(set_to_none=True)
                continue
            if use_amp:
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            if ema is not None and not vae_only:
                ema.update(model)
            if device.type == "cuda" and not vae_only and step % 50 == 0:
                reset_cuda_memory(device)
            running += loss_val
            postfix = {"loss": f"{running / step:.4f}"}
            if vae_only:
                postfix["vae"] = f"{float(cold_parts['vae_loss']):.4f}"
            else:
                postfix["hdr"] = f"{float(cold_parts['hdr_loss']):.4f}"
                postfix["cold"] = f"{float(cold_parts['cold_loss']):.4f}"
            pbar.set_postfix(postfix)

        train_loss = running / max(1, len(train_loader))
        epoch_time = time.time() - epoch_start

        model.eval()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        ema_backup = None
        if ema is not None and epoch > args.vae_warmup_epochs:
            ema_backup = ema.apply(model)

        tr_psnr, tr_ssim, tr_h2, tr_h3 = 0.0, 0.0, 0.0, 0.0
        probe_loader = _make_subset_loader(
            train_loader,
            max(1, args.train_eval_samples),
            args.num_workers,
            args.val_export_seed + epoch,
        )
        if probe_loader is not None:
            print(f"Train-probe ({len(probe_loader.dataset)} images)...")
            tr_psnr, tr_ssim, tr_h2, tr_h3 = validate_model_mtraining(
                probe_loader,
                device,
                epoch,
                hdrvdp_calculator,
                predict_fn,
                validation_root,
                save_samples=False,
                max_samples=0,
                amp=args.amp,
            )
            print(f"  Probe PSNR/SSIM: {tr_psnr:.4f} / {tr_ssim:.4f}")

        val_psnr, val_ssim, val_h2, val_h3 = 0.0, 0.0, 0.0, 0.0
        val_ran = False
        do_full = val_loader is not None and (
            epoch == 1
            or epoch % max(1, args.full_val_every) == 0
            or epoch == args.epochs
        )
        if do_full:
            print("Full validation...")
            val_psnr, val_ssim, val_h2, val_h3 = validate_model_mtraining(
                val_loader,
                device,
                epoch,
                hdrvdp_calculator,
                predict_fn,
                validation_root,
                save_samples=args.save_val_samples_each_epoch,
                max_samples=args.val_export_count,
                amp=args.amp,
            )
            val_ran = True
            print(f"  Val PSNR/SSIM: {val_psnr:.4f} / {val_ssim:.4f}")

        if ema_backup is not None:
            ema.restore(model, ema_backup)

        if scheduler is not None and epoch > args.vae_warmup_epochs:
            scheduler.step()
            print(f"  LR -> {optimizer.param_groups[0]['lr']:.2e}")

        metric_note = "full_val" if val_ran else f"train_probe_{args.train_eval_samples}"
        if ema is not None and epoch > args.vae_warmup_epochs:
            metric_note += ";ema_eval"
        if vae_only:
            metric_note += ";vae_warmup_eval=MonoLift_decode"
        _append_stage2_metrics_csv(
            csv_path,
            epoch,
            train_loss,
            tr_psnr,
            tr_ssim,
            tr_h2,
            tr_h3,
            val_psnr,
            val_ssim,
            val_h2,
            val_h3,
            val_ran=val_ran,
            vae_only=vae_only,
            metric_note=metric_note,
        )
        reported_psnr = val_psnr if val_ran else tr_psnr
        reported_ssim = val_ssim if val_ran else tr_ssim
        reported_h2 = val_h2 if val_ran else tr_h2
        reported_h3 = val_h3 if val_ran else tr_h3
        metric_src = "full val" if val_ran else f"train-probe ({args.train_eval_samples} imgs)"
        print_epoch_summary(
            epoch,
            args.epochs,
            train_loss,
            reported_psnr,
            reported_ssim,
            reported_h2,
            reported_h3,
            epoch_time,
        )
        if not val_ran:
            print(f"  (Epoch summary PSNR/SSIM are from {metric_src}, not full validation.)")
        if vae_only:
            print(
                f"  (Epoch {epoch}: VAE warmup — validation uses MonoLift+VAE decode, not full cold restore_hdr.)"
            )

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_probe_psnr": tr_psnr,
            "train_probe_ssim": tr_ssim,
            "val_psnr": val_psnr if val_ran else None,
            "val_ssim": val_ssim if val_ran else None,
            "best_val_psnr": best_psnr,
            "best_val_ssim": best_ssim,
            "timesteps": args.timesteps,
            "inference_timesteps": args.inference_timesteps,
            "base_ch": args.base_ch,
            "latent_ch": args.latent_ch,
            "vae_warmup_epochs": args.vae_warmup_epochs,
            "stage2_type": "cold_efficient_lorcd",
        }
        save_latest_checkpoint(args.checkpoint_dir, payload)

        if val_ran:
            if val_psnr > best_psnr:
                best_psnr, best_ssim = val_psnr, val_ssim
                best_hdrvdp2, best_hdrvdp3 = val_h2, val_h3
                full_val_streak = 0
                payload["best_val_psnr"] = best_psnr
                payload["best_val_ssim"] = best_ssim
                save_best_checkpoint(args.checkpoint_dir, payload)
                save_checkpoint(args.checkpoint_dir, f"best_epoch_{epoch}", payload)
            elif args.early_stop_patience > 0:
                full_val_streak += 1
                print(
                    f"  No full-val PSNR improvement ({full_val_streak}/{args.early_stop_patience}). "
                    f"Best={best_psnr:.4f} dB"
                )
                if full_val_streak >= args.early_stop_patience:
                    print(
                        f"[early_stop] Stopping at epoch {epoch}: "
                        f"no PSNR improvement for {args.early_stop_patience} full validations."
                    )
                    break

        if (
            epoch % args.save_ckpt_after == 0
            or epoch == args.epochs
            or epoch == args.vae_warmup_epochs
        ):
            save_checkpoint(args.checkpoint_dir, f"epoch_{epoch}", payload)
            print(f"  Saved epoch_{epoch}.pt")

        model.train()

    if not args.skip_final_test_export and val_idx:
        best_path = os.path.join(args.checkpoint_dir, "best.pt")
        if os.path.isfile(best_path):
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model"], strict=False)
            if ckpt.get("inference_timesteps") is not None:
                model.inference_timesteps = int(ckpt["inference_timesteps"])
            print(f"\nLoaded best checkpoint for final export (epoch {ckpt.get('epoch', '?')}).")
        model.eval()
        export_dir = args.val_export_dir or os.path.join(args.checkpoint_dir, "final_test_exports")
        export_final_test_samples(
            full_dataset,
            val_idx,
            device,
            make_stage2_epoch_predictor(model, vae_warmup_epochs=0),
            export_dir,
            count=args.final_test_count,
            seed=args.val_export_seed,
            amp=args.amp,
        )


if __name__ == "__main__":
    main()
