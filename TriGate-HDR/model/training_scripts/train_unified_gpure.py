"""
Unified GPURE training with full data validation (FHDR/test.py PSNR-μ + SSIM).

Phases:
  warmup — Stage-2 cold path (+ optional frozen Stage-1 for composition metrics)
  joint    — L_GPURE; on 20GB GPUs use --memory_20gb (freeze Stage-1, train Stage-2)
  seam     — Stage-3 seam refiner under unified energy

Metrics: identical to Stage 2 / legacy TriGate — validate_model_mtraining +
compute_psnr_ssim (FHDR/test.py, no sanitization before metrics).
"""

from __future__ import annotations

import argparse
import csv
import os
import time

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..decoders.cold_hdr_diffusion_decoder import ColdHDRDiffusion
from ..dual_decoders.cold_hdr_luminance_diffusion_decoder import Stage1TriEncoderDiffusionSystem
from ..seaming_model.gan_system import SeamingGANSystem
from ..unified.gpure_energy import GPUREEnergyConfig
from ..unified.trigate_composer import TriGateComposer
from ..unified.trigate_gpure_system import TriGateGPURESystem
from .common_training import (
    HDRVDPMetrics,
    add_subset_args,
    apply_smoke_test_args,
    default_hrishav_data_paths,
    maybe_resume,
    print_stage2_epoch_summary,
    reset_cuda_memory,
    sanitize_data_path,
    save_best_checkpoint,
    save_checkpoint,
    save_latest_checkpoint,
)
from .dataset_splits import build_dataloaders
from .val_export import export_final_test_samples, make_gpure_predictor, validate_model_mtraining


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


def _append_gpure_metrics_csv(
    csv_path,
    epoch,
    train_loss,
    tr_psnr,
    tr_ssim,
    val_psnr,
    val_ssim,
    val_full_psnr,
    val_full_ssim,
    phase: str,
    metric_note: str = "",
    tr_hdrvdp2: float = 0.0,
    tr_hdrvdp3: float = 0.0,
    val_hdrvdp2: float = 0.0,
    val_hdrvdp3: float = 0.0,
    val_full_hdrvdp2=None,
    val_full_hdrvdp3=None,
):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        fieldnames = [
            "epoch",
            "phase",
            "train_loss",
            "train_psnr",
            "train_ssim",
            "train_hdrvdp2",
            "train_hdrvdp3",
            "val_psnr",
            "val_ssim",
            "val_hdrvdp2",
            "val_hdrvdp3",
            "val_full_psnr",
            "val_full_ssim",
            "val_full_hdrvdp2",
            "val_full_hdrvdp3",
            "metric_note",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": epoch,
                "phase": phase,
                "train_loss": f"{float(train_loss):.6f}",
                "train_psnr": f"{float(tr_psnr):.4f}",
                "train_ssim": f"{float(tr_ssim):.4f}",
                "train_hdrvdp2": f"{float(tr_hdrvdp2):.4f}",
                "train_hdrvdp3": f"{float(tr_hdrvdp3):.4f}",
                "val_psnr": f"{float(val_psnr):.4f}",
                "val_ssim": f"{float(val_ssim):.4f}",
                "val_hdrvdp2": f"{float(val_hdrvdp2):.4f}",
                "val_hdrvdp3": f"{float(val_hdrvdp3):.4f}",
                "val_full_psnr": f"{float(val_full_psnr):.4f}" if val_full_psnr is not None else "",
                "val_full_ssim": f"{float(val_full_ssim):.4f}" if val_full_ssim is not None else "",
                "val_full_hdrvdp2": f"{float(val_full_hdrvdp2):.4f}" if val_full_hdrvdp2 is not None else "",
                "val_full_hdrvdp3": f"{float(val_full_hdrvdp3):.4f}" if val_full_hdrvdp3 is not None else "",
                "metric_note": metric_note,
            }
        )
    print(f"  Metrics CSV updated: {csv_path}")


def _apply_memory_20gb(args):
    """Preset for ~20GB VRAM: fits Stage-2 training + frozen Stage-1 inference + validation."""
    args.batch_size = 1
    if args.max_dim <= 0 or args.max_dim > 512:
        args.max_dim = 512
    args.amp = True
    args.freeze_stage1 = True
    args.num_workers = min(args.num_workers, 2)
    if args.inference_timesteps > 25:
        args.inference_timesteps = 25
    if args.stage1_inference_steps > 10:
        args.stage1_inference_steps = 10
    print(
        "[memory_20gb] batch=1 max_dim=512 amp=True freeze_stage1=True "
        f"inference_timesteps={args.inference_timesteps} stage1_steps={args.stage1_inference_steps}"
    )


def _apply_arch_v2(args):
    args.base_ch = 96
    args.latent_ch = 8
    args.vae_base_ch = 48
    args.use_pixel_refiner = True
    print("[arch_v2] Wider VAE/UNet + pixel refiner enabled.")


def _build_stage1(device: torch.device, init_ckpt: str = ""):
    model = Stage1TriEncoderDiffusionSystem().to(device)
    if init_ckpt and os.path.isfile(init_ckpt):
        ckpt = torch.load(init_ckpt, map_location="cpu")
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        print(f"[GPURE] Loaded Stage-1 weights from {init_ckpt}")
    return model


def _build_stage2(device: torch.device, args) -> ColdHDRDiffusion:
    model = ColdHDRDiffusion(
        timesteps=args.timesteps,
        base_ch=args.base_ch,
        latent_ch=args.latent_ch,
        vae_base_ch=args.vae_base_ch,
        use_pixel_refiner=args.use_pixel_refiner,
        use_rso=args.use_rso,
        use_lr_cfp=args.use_lr_cfp,
    ).to(device)
    if args.init_stage2 and os.path.isfile(args.init_stage2):
        ckpt = torch.load(args.init_stage2, map_location="cpu")
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        if ckpt.get("inference_timesteps") is not None:
            model.inference_timesteps = int(ckpt["inference_timesteps"])
        print(f"[GPURE] Loaded Stage-2 weights from {args.init_stage2}")
    model.inference_timesteps = int(args.inference_timesteps)
    return model


def _build_gpure_system(device: torch.device, args) -> TriGateGPURESystem:
    stage1 = _build_stage1(device, args.init_stage1)
    stage2 = _build_stage2(device, args)
    stage3 = None
    if args.phase == "seam":
        gan = SeamingGANSystem(use_rso_stem=args.use_rso).to(device)
        stage3 = gan.generator
        if args.init_stage3 and os.path.isfile(args.init_stage3):
            ckpt = torch.load(args.init_stage3, map_location="cpu")
            gan.load_state_dict(ckpt.get("model", ckpt), strict=False)
            print(f"[GPURE] Loaded Stage-3 weights from {args.init_stage3}")
    energy_cfg = GPUREEnergyConfig(
        lambda_rad=args.lambda_rad,
        lambda_cold=args.lambda_cold,
        lambda_gen=args.lambda_gen,
        lambda_bracket=args.lambda_bracket,
        lambda_seam=args.lambda_seam,
    )
    composer = TriGateComposer(soft_seam_gamma=args.soft_seam_gamma)
    return TriGateGPURESystem(
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        composer=composer,
        energy_cfg=energy_cfg,
        use_stage3=(args.phase == "seam"),
        stage1_inference_steps=args.stage1_inference_steps,
        stage2_inference_steps=args.inference_timesteps,
        freeze_stage1=args.freeze_stage1,
    ).to(device)


def _set_trainable(system: TriGateGPURESystem, phase: str, freeze_stage1: bool) -> None:
    for p in system.parameters():
        p.requires_grad = False
    if phase == "warmup":
        for p in system.stage2.parameters():
            p.requires_grad = True
    elif phase == "joint":
        for p in system.stage2.parameters():
            p.requires_grad = True
        if not freeze_stage1:
            for p in system.stage1.parameters():
                p.requires_grad = True
    elif phase == "seam" and system.stage3 is not None:
        for p in system.stage3.parameters():
            p.requires_grad = True


def main():
    defaults = default_hrishav_data_paths()
    parser = argparse.ArgumentParser(
        description="Unified GPURE training (TriGate-HDR) with FHDR metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phase", type=str, choices=["warmup", "joint", "seam"], default="joint")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_dim", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--inference_timesteps", type=int, default=50)
    parser.add_argument("--stage1_inference_steps", type=int, default=25)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--latent_ch", type=int, default=4)
    parser.add_argument("--vae_base_ch", type=int, default=32)
    parser.add_argument("--use_pixel_refiner", action="store_true")
    parser.add_argument("--arch_v2", action="store_true")
    parser.add_argument("--use_rso", action="store_true")
    parser.add_argument("--use_lr_cfp", action="store_true")
    parser.add_argument("--soft_seam_gamma", type=float, default=0.0)
    parser.add_argument("--lambda_rad", type=float, default=1.0)
    parser.add_argument("--lambda_cold", type=float, default=1.0)
    parser.add_argument("--lambda_gen", type=float, default=1.0)
    parser.add_argument("--lambda_bracket", type=float, default=0.5)
    parser.add_argument("--lambda_seam", type=float, default=0.25)
    parser.add_argument("--init_stage1", type=str, default="")
    parser.add_argument("--init_stage2", type=str, default="")
    parser.add_argument("--init_stage3", type=str, default="")
    parser.add_argument("--checkpoint_dir", type=str, default="experiments/gpure_unified")
    parser.add_argument("--ldr_dir", type=str, default=defaults["ldr_dir"])
    parser.add_argument("--hdr_dir", type=str, default=defaults["hdr_dir"])
    parser.add_argument("--sam_mask_dir", type=str, default=defaults.get("sam_mask_dir", ""))
    parser.add_argument("--max_sam_classes", type=int, default=64)
    parser.add_argument(
        "--freeze_stage1",
        action="store_true",
        help="Stage-1 inference only (no grad). Required for ~20GB joint training.",
    )
    parser.add_argument(
        "--memory_20gb",
        action="store_true",
        help="Preset: batch=1, max_dim=512, amp, freeze_stage1, fewer inference steps.",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--continue_train", action="store_true")
    parser.add_argument("--resume_best", action="store_true")
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--full_val_every", type=int, default=10)
    parser.add_argument("--train_eval_samples", type=int, default=50)
    parser.add_argument("--val_eval_samples", type=int, default=50)
    parser.add_argument("--trial_val_samples", type=int, default=5)
    parser.add_argument("--skip_trial_validation", action="store_true")
    parser.add_argument("--hdrvdp_fast_proxy", action="store_true")
    add_subset_args(parser)
    args = parser.parse_args()
    args = apply_smoke_test_args(args)

    if args.arch_v2:
        _apply_arch_v2(args)
    if args.memory_20gb:
        _apply_memory_20gb(args)

    args.ldr_dir = sanitize_data_path(args.ldr_dir)
    args.hdr_dir = sanitize_data_path(args.hdr_dir)
    args.sam_mask_dir = sanitize_data_path(args.sam_mask_dir) if args.sam_mask_dir else ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    validation_root = args.validation_results_dir or os.path.join(args.checkpoint_dir, "validation_results")
    os.makedirs(validation_root, exist_ok=True)
    csv_path = os.path.join(args.checkpoint_dir, "training_metrics.csv")

    print(
        f"[GPURE] phase={args.phase} freeze_stage1={args.freeze_stage1} "
        f"use_rso={args.use_rso} use_lr_cfp={args.use_lr_cfp} memory_20gb={args.memory_20gb}"
    )

    system = _build_gpure_system(device, args)
    _set_trainable(system, args.phase, args.freeze_stage1)

    params = [p for p in system.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr)
    scaler = GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    hdrvdp_calculator = HDRVDPMetrics(use_real_hdrvdp=not args.hdrvdp_fast_proxy)
    predict_fn = make_gpure_predictor(system)

    start_epoch = 1
    best_psnr, best_ssim = 0.0, 0.0
    best_hdrvdp2, best_hdrvdp3 = 0.0, 0.0
    full_val_streak = 0

    if args.continue_train:
        start_epoch, best_psnr, best_ssim, best_hdrvdp2, best_hdrvdp3 = maybe_resume(
            args.checkpoint_dir,
            system,
            optimizer,
            resume_from=sanitize_data_path(args.resume_from) if args.resume_from else "",
            strict=False,
            prefer_best=args.resume_best,
        )
        system = system.to(device)
        if device.type == "cuda":
            reset_cuda_memory(device, "after resume")

    train_loader, val_loader, full_dataset, val_idx = build_dataloaders(
        args.ldr_dir,
        args.hdr_dir,
        args.batch_size,
        sam_mask_dir=args.sam_mask_dir,
        max_sam_classes=args.max_sam_classes,
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
        f"[GPURE] train={len(train_loader.dataset)} val={len(val_loader.dataset) if val_loader else 0} "
        f"batch_size={args.batch_size} max_dim={args.max_dim}"
    )
    print(f"  Metrics (FHDR/test.py PSNR-μ + SSIM): {csv_path}")

    if start_epoch == 1 and not args.skip_trial_validation and val_loader is not None:
        trial_loader = _make_subset_loader(
            val_loader, max(1, args.trial_val_samples), args.num_workers, args.val_export_seed
        )
        if trial_loader is not None:
            print("\n" + "=" * 60)
            print(f"Trial validation before epoch 1 ({len(trial_loader.dataset)} images)")
            print("=" * 60)
            system.eval()
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
            _append_gpure_metrics_csv(
                csv_path,
                epoch=0,
                train_loss=0.0,
                tr_psnr=0.0,
                tr_ssim=0.0,
                val_psnr=t_psnr,
                val_ssim=t_ssim,
                val_full_psnr=t_psnr,
                val_full_ssim=t_ssim,
                phase=args.phase,
                metric_note="trial_val_before_epoch_1",
                val_hdrvdp2=t_h2,
                val_hdrvdp3=t_h3,
                val_full_hdrvdp2=t_h2,
                val_full_hdrvdp3=t_h3,
            )
            print("=" * 60 + "\n")
            system.train()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        system.train()
        running = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"GPURE-{args.phase} {epoch}/{args.epochs}"):
            ldr = batch["ldr_image"].to(device)
            hdr = batch["hdr_image"].to(device)
            gate = batch.get("gate")
            if gate is not None:
                gate = gate.to(device)
            batch_dev = {
                "ldr_image": ldr,
                "hdr_image": hdr,
                "gate": gate,
                "segmap": batch.get("segmap"),
            }

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=args.amp and device.type == "cuda"):
                outputs = system(batch_dev, mode="train")
                energy = system.compute_energy(outputs, batch_dev)
                loss = energy.total
                if args.phase == "warmup":
                    s2 = outputs.stage2_parts.get("loss")
                    if s2 is not None:
                        loss = s2

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.detach().cpu())
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        epoch_time = time.time() - epoch_start

        system.eval()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        tr_psnr, tr_ssim, tr_h2, tr_h3 = 0.0, 0.0, 0.0, 0.0
        train_probe = _make_subset_loader(
            train_loader,
            max(1, args.train_eval_samples),
            args.num_workers,
            args.val_export_seed + epoch,
        )
        if train_probe is not None:
            print(f"Train metrics ({len(train_probe.dataset)} images)...")
            tr_psnr, tr_ssim, tr_h2, tr_h3 = validate_model_mtraining(
                train_probe,
                device,
                epoch,
                hdrvdp_calculator,
                predict_fn,
                validation_root,
                save_samples=False,
                max_samples=0,
                amp=args.amp,
            )
            print(f"  Train PSNR/SSIM/H2/H3: {tr_psnr:.4f} / {tr_ssim:.4f} / {tr_h2:.4f} / {tr_h3:.4f}")

        val_psnr, val_ssim, val_h2, val_h3 = 0.0, 0.0, 0.0, 0.0
        val_full_psnr, val_full_ssim, val_full_h2, val_full_h3 = None, None, None, None
        if val_loader is not None:
            val_subset = _make_subset_loader(
                val_loader,
                max(1, args.val_eval_samples),
                args.num_workers,
                args.val_export_seed + epoch + 10000,
            )
            if val_subset is not None:
                print(f"Val metrics ({len(val_subset.dataset)} images)...")
                val_psnr, val_ssim, val_h2, val_h3 = validate_model_mtraining(
                    val_subset,
                    device,
                    epoch,
                    hdrvdp_calculator,
                    predict_fn,
                    validation_root,
                    save_samples=False,
                    max_samples=0,
                    amp=args.amp,
                )
                print(f"  Val PSNR/SSIM/H2/H3: {val_psnr:.4f} / {val_ssim:.4f} / {val_h2:.4f} / {val_h3:.4f}")

        do_full = val_loader is not None and (
            epoch == 1 or epoch % max(1, args.full_val_every) == 0 or epoch == args.epochs
        )
        if do_full:
            print(f"Full validation ({len(val_loader.dataset)} images)...")
            val_full_psnr, val_full_ssim, val_full_h2, val_full_h3 = validate_model_mtraining(
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
            print(
                f"  Val-Full PSNR/SSIM/H2/H3: {val_full_psnr:.4f} / {val_full_ssim:.4f} "
                f"/ {val_full_h2:.4f} / {val_full_h3:.4f}"
            )

        score_psnr = val_full_psnr if val_full_psnr is not None else val_psnr
        metric_note = f"phase={args.phase};train_n={args.train_eval_samples};val_n={args.val_eval_samples}"
        if val_full_psnr is not None:
            metric_note += ";val_full"
        _append_gpure_metrics_csv(
            csv_path,
            epoch,
            train_loss,
            tr_psnr,
            tr_ssim,
            val_psnr,
            val_ssim,
            val_full_psnr,
            val_full_ssim,
            phase=args.phase,
            metric_note=metric_note,
            tr_hdrvdp2=tr_h2,
            tr_hdrvdp3=tr_h3,
            val_hdrvdp2=val_h2,
            val_hdrvdp3=val_h3,
            val_full_hdrvdp2=val_full_h2,
            val_full_hdrvdp3=val_full_h3,
        )
        print_stage2_epoch_summary(
            epoch,
            args.epochs,
            train_loss,
            tr_psnr,
            tr_ssim,
            val_psnr,
            val_ssim,
            val_full_psnr,
            val_full_ssim,
            epoch_time,
            train_hdrvdp2=tr_h2,
            train_hdrvdp3=tr_h3,
            val_hdrvdp2=val_h2,
            val_hdrvdp3=val_h3,
            val_full_hdrvdp2=val_full_h2,
            val_full_hdrvdp3=val_full_h3,
        )

        payload = {
            "epoch": epoch,
            "model": system.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "val_full_psnr": val_full_psnr,
            "val_full_ssim": val_full_ssim,
            "best_val_psnr": best_psnr,
            "best_val_ssim": best_ssim,
            "phase": args.phase,
            "use_rso": args.use_rso,
            "use_lr_cfp": args.use_lr_cfp,
            "freeze_stage1": args.freeze_stage1,
            "inference_timesteps": args.inference_timesteps,
            "stage1_type": "tri_encoder_legacy",
            "stage2_type": "cold_efficient_lorcd_gpure",
        }
        save_latest_checkpoint(args.checkpoint_dir, payload)

        if score_psnr > best_psnr:
            best_psnr = score_psnr
            best_ssim = val_full_ssim if val_full_ssim is not None else val_ssim
            full_val_streak = 0
            payload["best_val_psnr"] = best_psnr
            payload["best_val_ssim"] = best_ssim
            save_best_checkpoint(args.checkpoint_dir, payload)
            save_checkpoint(args.checkpoint_dir, f"best_epoch_{epoch}", payload)
            print(f"  New best.pt (PSNR={best_psnr:.4f} dB)")
        elif args.early_stop_patience > 0:
            full_val_streak += 1
            print(f"  No val PSNR improvement ({full_val_streak}/{args.early_stop_patience})")
            if full_val_streak >= args.early_stop_patience:
                print(f"Early stop at epoch {epoch}. Best PSNR={best_psnr:.4f} dB")
                break

        system.train()

    if not args.skip_final_test_export and val_idx:
        export_dir = args.val_export_dir or os.path.join(args.checkpoint_dir, "final_test_export")
        print(f"\nFinal test export -> {export_dir}")
        export_final_test_samples(
            full_dataset,
            val_idx,
            device,
            predict_fn,
            export_dir,
            count=args.final_test_count,
            seed=args.val_export_seed,
            amp=args.amp,
        )


if __name__ == "__main__":
    main()
