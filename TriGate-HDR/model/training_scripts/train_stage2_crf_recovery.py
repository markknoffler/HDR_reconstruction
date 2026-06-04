"""
Train Stage-2 ColdEfficient-LORCD (latent cold diffusion, no foundation model).

Validation metrics: FHDR/test.py PSNR-μ and SSIM via common_training.compute_psnr_ssim.
"""

import argparse
import os
import time

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
    add_subset_args,
    apply_smoke_test_args,
    default_hrishav_data_paths,
    maybe_resume,
    print_epoch_summary,
    save_best_checkpoint,
    save_checkpoint,
    save_latest_checkpoint,
    save_metrics_to_csv,
    sanitize_data_path,
)
from .dataset_splits import build_dataloaders
from .val_export import make_stage2_predictor, validate_model_mtraining


def _make_subset_loader(loader, sample_count: int, num_workers: int, seed: int):
    ds = loader.dataset
    indices = list(getattr(ds, "indices", []))
    if not indices:
        return None
    if len(indices) <= sample_count:
        picked = indices
    else:
        g = torch.Generator()
        g.manual_seed(int(seed))
        order = torch.randperm(len(indices), generator=g).tolist()
        picked = [indices[i] for i in order[:sample_count]]
    subset = Subset(ds.dataset, picked)
    return DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def main():
    _defaults = default_hrishav_data_paths()
    parser = argparse.ArgumentParser(
        description="Train Stage-2 ColdEfficient-LORCD (latent expansion cold diffusion).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
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
        default=25,
        help="Cold reverse steps for restore_hdr at validation (lower saves VRAM/time).",
    )
    parser.add_argument("--base_ch", type=int, default=64, help="Latent UNet base channels.")
    parser.add_argument("--latent_ch", type=int, default=4, help="MiniHDR-VAE latent channels.")
    parser.add_argument("--vae_warmup_epochs", type=int, default=5, help="VAE-only warmup before cold training.")
    parser.add_argument("--cold_loss_weight", type=float, default=1.0)
    parser.add_argument("--exp_loss_weight", type=float, default=1.0)
    parser.add_argument("--trust_loss_weight", type=float, default=0.5)
    parser.add_argument("--ms_cold_weight", type=float, default=0.25)
    parser.add_argument("--vae_loss_weight", type=float, default=0.1)
    parser.add_argument("--mono_loss_weight", type=float, default=0.01)
    parser.add_argument("--radiometric_weight", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--continue_train", action="store_true")
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
    if args.continue_train:
        start_epoch, best_psnr, best_ssim, best_hdrvdp2, best_hdrvdp3 = maybe_resume(
            args.checkpoint_dir, model, optimizer, strict=False
        )
        model = model.to(device)
        resume_path = os.path.join(args.checkpoint_dir, "latest.pt")
        if os.path.isfile(resume_path):
            ckpt_meta = torch.load(resume_path, map_location="cpu")
            if ckpt_meta.get("inference_timesteps") is not None:
                model.inference_timesteps = int(ckpt_meta["inference_timesteps"])
        print(
            f"Resuming from epoch {start_epoch} (best PSNR={best_psnr:.4f}), "
            f"inference_timesteps={model.inference_timesteps}"
        )

    train_loader, val_loader, _, _ = build_dataloaders(
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
    print(
        f"[Stage2] train={len(train_loader.dataset)} val={len(val_loader.dataset) if val_loader else 0} "
        f"smoke_test={args.smoke_test}"
    )
    print(f"  Metrics (FHDR/test.py PSNR-μ + SSIM): {csv_path}")
    print(f"  Trial val before epoch 1: {not args.skip_trial_validation and start_epoch == 1}")

    predict_fn = make_stage2_predictor(model)

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
            print("=" * 60 + "\n")
            model.train()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Stage2 Cold {epoch}/{args.epochs}", leave=True)
        vae_only = epoch <= args.vae_warmup_epochs
        if epoch == args.vae_warmup_epochs + 1 and device.type == "cuda":
            torch.cuda.empty_cache()
            print("[Stage2-LORCD] VAE warmup done — cleared CUDA cache before full cold training.")
        for step, batch in enumerate(pbar, start=1):
            ldr = batch["ldr_image"].to(device)
            hdr = batch["hdr_image"].to(device)
            gate = batch["gate"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=args.amp and device.type == "cuda"):
                pred, cold_parts = model(hdr, ldr, gate=gate, vae_only=vae_only)
                if vae_only:
                    loss = args.vae_loss_weight * cold_parts["vae_loss"]
                else:
                    loss = (
                        cold_parts["hdr_loss"]
                        + args.cold_loss_weight * cold_parts["cold_loss"]
                        + args.exp_loss_weight * cold_parts["exp_loss"]
                        + args.trust_loss_weight * cold_parts["trust_loss"]
                        + args.ms_cold_weight * cold_parts["ms_cold_loss"]
                        + args.mono_loss_weight * cold_parts["mono_loss"]
                        + args.vae_loss_weight * cold_parts["vae_loss"]
                    )
                    pred_lin = (pred + 1.0) * 0.5
                    hdr_lin = (hdr + 1.0) * 0.5
                    rad_loss, _ = stage2_loss(
                        pred_lin, hdr_lin, ldr, gate, radiometric_loss_fn=radiometric_loss_fn
                    )
                    loss = loss + args.radiometric_weight * rad_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda" and not vae_only and step % 50 == 0:
                torch.cuda.empty_cache()
            running += loss.item()
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
            epoch % max(1, args.full_val_every) == 0 or epoch == args.epochs
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

        save_metrics_to_csv(
            csv_path,
            epoch,
            train_loss,
            val_psnr if val_ran else tr_psnr,
            val_ssim if val_ran else tr_ssim,
            val_h2 if val_ran else tr_h2,
            val_h3 if val_ran else tr_h3,
        )
        print_epoch_summary(
            epoch,
            args.epochs,
            train_loss,
            val_psnr if val_ran else tr_psnr,
            val_ssim if val_ran else tr_ssim,
            val_h2 if val_ran else tr_h2,
            val_h3 if val_ran else tr_h3,
            epoch_time,
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

        if val_ran and val_psnr > best_psnr:
            best_psnr, best_ssim = val_psnr, val_ssim
            best_hdrvdp2, best_hdrvdp3 = val_h2, val_h3
            payload["best_val_psnr"] = best_psnr
            payload["best_val_ssim"] = best_ssim
            save_best_checkpoint(args.checkpoint_dir, payload)
            save_checkpoint(args.checkpoint_dir, f"best_epoch_{epoch}", payload)

        if epoch % args.save_ckpt_after == 0 or epoch == args.epochs:
            save_checkpoint(args.checkpoint_dir, f"epoch_{epoch}", payload)
            print(f"  Saved epoch_{epoch}.pt")

        model.train()


if __name__ == "__main__":
    main()
