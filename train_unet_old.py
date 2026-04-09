"""
Training script for the UNet denoiser with S1+S2 fusion conditioning.

Usage:
    python train_unet.py --data_dir /path/to/npz_tiles --vae_ckpt /path/to/vae.ckpt

Prerequisites:
    - A pretrained VAE (AutoencoderKL) checkpoint trained on aerial RGBNIR data.
    - NPZ tiles from the Bachelor pipeline containing s1, s2, aerial, valid arrays.

Training flow per step (5 m aerial, scale_factor=2, padded 128→256):
    1. aerial (B,4,256,256) → normalize [-1,1] → VAE encode → z_0 (B,4,64,64)
    2. S2 (B,4,128,128) → normalize → upsample 256 → VAE encode → cond_s2 (B,4,64,64)
       S1 (B,2,128,128) → normalize → upsample 64             → cond_s1 (B,2,64,64)
       conditioning = concat(cond_s2, cond_s1) → (B,6,64,64)
    3. Sample t ~ Uniform(0, T), eps ~ N(0, I)
    4. z_t = sqrt(α̅_t) · z_0 + sqrt(1−α̅_t) · eps
    5. eps_pred = UNet(concat(z_t, conditioning), t)
    6. loss = MSE(eps_pred, eps)
"""

import argparse
import pathlib
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from opensr_model.data import FusionDataset, make_train_val_datasets
from opensr_model.diffusion.latentdiffusion import LatentDiffusion
from opensr_model.utils import normalize_s1, normalize_s2, normalize_aerial


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_model(config, device):
    """Instantiate LatentDiffusion from the YAML config."""
    model = LatentDiffusion(
        first_stage_config=dict(config.first_stage_config),
        cond_stage_config="__is_first_stage__",
        cond_stage_key=config.other.cond_stage_key,
        first_stage_key=config.other.first_stage_key,
        cond_stage_trainable=config.other.cond_stage_trainable,
        concat_mode=config.other.concat_mode,
        unet_config=dict(config.cond_stage_config),
        timesteps=config.denoiser_settings.timesteps,
        linear_start=config.denoiser_settings.linear_start,
        linear_end=config.denoiser_settings.linear_end,
        parameterization=config.denoiser_settings.parameterization,
    )
    model = model.to(device)
    return model


def load_vae_weights(model, vae_ckpt_path, device):
    """Load pretrained VAE weights into model.first_stage_model."""
    print(f"Loading VAE weights from: {vae_ckpt_path}")
    state_dict = torch.load(vae_ckpt_path, map_location=device)

    # Handle different checkpoint formats
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Strip 'module.' prefix from DataParallel checkpoints
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        cleaned[k] = v

    model.first_stage_model.load_state_dict(cleaned, strict=False)
    print(f"  VAE loaded ({len(cleaned)} keys)")


def build_conditioning(s2, s1, model, config, device):
    """Build 6-channel fused conditioning tensor in latent space.

    Same logic as SRLatentDiffusion._tensor_encode but standalone for training.

    Args:
        s2: (B, 4, 128, 128) raw S2 DN values
        s1: (B, 2, 128, 128) raw S1 dB values
        model: LatentDiffusion instance (for VAE encoder)
        config: OmegaConf config
        device: torch device

    Returns:
        conditioning: (B, 6, latent_H, latent_W)
    """
    scale_factor = config.scale_factor
    ch_mult = list(config.first_stage_config.ch_mult)
    vae_downscale = 2 ** (len(ch_mult) - 1)

    lr_size = s2.shape[-1]                     # 128
    hr_size = lr_size * scale_factor            # 1280
    latent_size = hr_size // vae_downscale      # 320

    # S2: normalize → upsample to HR → VAE encode → 4ch latent
    s2_norm = normalize_s2(s2.to(device), stage="norm")
    s2_up = F.interpolate(s2_norm, size=(hr_size, hr_size), mode='bilinear', align_corners=False)
    with torch.no_grad():
        cond_s2 = model.first_stage_model.encode(s2_up).sample()

    # S1: normalize → upsample to latent size → 2ch
    s1_norm = normalize_s1(s1.to(device), stage="norm")
    cond_s1 = F.interpolate(s1_norm, size=(latent_size, latent_size), mode='bilinear', align_corners=False)

    # Concat → 6ch conditioning
    conditioning = torch.cat([cond_s2, cond_s1], dim=1)
    return conditioning


def encode_aerial_target(aerial, model, device):
    """Encode aerial HR image into VAE latent z_0.

    Args:
        aerial: (B, 4, 1280, 1280) raw uint8 values as float
        model: LatentDiffusion instance
        device: torch device

    Returns:
        z_0: (B, 4, 320, 320) scaled latent
    """
    aerial_norm = normalize_aerial(aerial.to(device), stage="norm")
    with torch.no_grad():
        posterior = model.encode_first_stage(aerial_norm)
        z_0 = model.get_first_stage_encoding(posterior)  # includes scale_factor
    return z_0


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, config, device, epoch):
    """Train UNet for one epoch. Returns average loss."""
    model.model.train()  # DiffusionWrapper (UNet) in train mode
    # VAE stays in eval (frozen by LatentDiffusion.__init__)

    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        s1 = batch["s1"]         # (B, 2, 128, 128)
        s2 = batch["s2"]         # (B, 4, 128, 128)
        aerial = batch["aerial"] # (B, 4, 1280, 1280)

        # 1. Encode aerial target → z_0
        z_0 = encode_aerial_target(aerial, model, device)

        # 2. Build 6ch conditioning from S1+S2
        conditioning = build_conditioning(s2, s1, model, config, device)

        # 3. Sample random timestep t for each item in batch
        B = z_0.shape[0]
        t = torch.randint(0, model.num_timesteps, (B,), device=device).long()

        # 4. Sample noise and create noisy latent
        noise = torch.randn_like(z_0)
        z_t = model.q_sample(x_start=z_0, t=t, noise=noise)

        # 5. UNet predicts noise from (z_t concat conditioning)
        noise_pred = model.apply_model(z_t, t, cond=conditioning)

        # 6. Loss = MSE(predicted noise, actual noise)
        loss = F.mse_loss(noise_pred, noise)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 50 == 0:
            print(f"  [Epoch {epoch}] Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.6f}")

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, config, device):
    """Validate UNet. Returns average loss."""
    model.model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        s1 = batch["s1"]
        s2 = batch["s2"]
        aerial = batch["aerial"]

        z_0 = encode_aerial_target(aerial, model, device)
        conditioning = build_conditioning(s2, s1, model, config, device)

        B = z_0.shape[0]
        t = torch.randint(0, model.num_timesteps, (B,), device=device).long()
        noise = torch.randn_like(z_0)
        z_t = model.q_sample(x_start=z_0, t=t, noise=noise)

        noise_pred = model.apply_model(z_t, t, cond=conditioning)
        loss = F.mse_loss(noise_pred, noise)

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train UNet denoiser with S1+S2 fusion")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to NPZ tile directory")
    parser.add_argument("--vae_ckpt", type=str, required=True, help="Path to pretrained VAE checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config (default: config_10m.yaml)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/unet", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    # min_valid_frac removed — NPZ tiles no longer contain a 'valid' mask
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--use_mlflow", action="store_true", help="Log metrics to MLflow")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ── Config ──────────────────────────────────────────────────────────────
    if args.config is None:
        config_path = pathlib.Path(__file__).parent / "opensr_model" / "configs" / "config_10m.yaml"
    else:
        config_path = pathlib.Path(args.config)
    config = OmegaConf.load(config_path)
    print(f"Config loaded from: {config_path}")

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Output dir ──────────────────────────────────────────────────────────
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ───────────────────────────────────────────────────────────────
    print("Building LatentDiffusion model...")
    model = build_model(config, device)

    # Load pretrained VAE
    load_vae_weights(model, args.vae_ckpt, device)

    # Verify: VAE frozen, UNet trainable
    vae_params = sum(p.numel() for p in model.first_stage_model.parameters())
    vae_trainable = sum(p.numel() for p in model.first_stage_model.parameters() if p.requires_grad)
    unet_params = sum(p.numel() for p in model.model.parameters())
    unet_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"VAE:  {vae_params:,} params ({vae_trainable:,} trainable) — should be 0 trainable")
    print(f"UNet: {unet_params:,} params ({unet_trainable:,} trainable)")

    # ── Data ────────────────────────────────────────────────────────────────
    print(f"Loading data from: {args.data_dir}")
    train_ds, val_ds = make_train_val_datasets(
        root=args.data_dir,
        val_frac=args.val_frac,
    )
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Optimizer ───────────────────────────────────────────────────────────
    # Only optimize UNet (DiffusionWrapper) parameters
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── MLflow (optional) ───────────────────────────────────────────────────
    if args.use_mlflow:
        import mlflow
        mlflow.set_experiment("unet-s1s2-fusion")
        mlflow.start_run(run_name=f"unet_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "min_valid_frac": args.min_valid_frac,
            "scale_factor": config.scale_factor,
            "unet_in_channels": config.cond_stage_config.in_channels,
            "timesteps": config.denoiser_settings.timesteps,
            "parameterization": config.denoiser_settings.parameterization,
        })

    # ── Training ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Starting training: {args.epochs} epochs, batch_size={args.batch_size}")
    print(f"Train: {len(train_ds)} tiles, Val: {len(val_ds)} tiles")
    print(f"{'='*60}\n")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(model, train_dl, optimizer, config, device, epoch)

        # Validate
        val_loss = validate(model, val_dl, config, device)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s")

        # MLflow logging
        if args.use_mlflow:
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
            }, step=epoch)

        # Save checkpoint
        if epoch % args.save_every == 0 or val_loss < best_val_loss:
            ckpt_path = output_dir / f"unet_epoch{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "unet_state_dict": model.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_dir / "unet_best.pt"
                torch.save({
                    "epoch": epoch,
                    "unet_state_dict": model.model.state_dict(),
                    "val_loss": val_loss,
                }, best_path)
                print(f"  ★ New best model! Val loss: {val_loss:.6f}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")

    if args.use_mlflow:
        mlflow.log_artifact(str(output_dir / "unet_best.pt"))
        mlflow.end_run()


if __name__ == "__main__":
    main()
