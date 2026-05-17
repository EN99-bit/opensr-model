"""
Training script for a 5m→1m UNet denoiser (cascade stage 2, with S1 + S2 conditioning).

Variant of train_unet_5-1_with_s1.py that additionally conditions on the original S2
satellite input alongside the 5m aerial and S1 SAR.

Conditioning: 5m aerial VAE (4ch) + S1 (2ch) + S2 VAE (4ch) = 10ch total
Target:       1m aerial (1024×1024), encoded with the 1m VAE.

Config:       config_5m_to_1m_with_s2.yaml  (in_channels: 14 = 4 noise + 10 cond)

Usage:
    python train_unet_5-1_with_s1_s2.py \
        --data_dir_5m ~/npz/apr2025/5m-npz \
        --data_dir_1m ~/npz/apr2025/1m-npz \
        --vae_ckpt checkpoints/1m/vae/vae.ckpt
"""

import argparse
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.tuner import Tuner

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from opensr_model.diffusion.latentdiffusion import LatentDiffusion
from opensr_model.utils import normalize_aerial, normalize_s1, normalize_s2

# Padded spatial sizes
LR_PAD_SIZE = 128          # S1/S2: 100×100 native → 128×128
AERIAL_5M_PAD_SIZE = 256   # 5m aerial: 200×200 native → 256×256
HR_PAD_SIZE = 1024         # 1m aerial: 1000×1000 native → 1024×1024


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class Fusion5mTo1mDataset(Dataset):
    """Paired dataset loading 5m aerial + S1 + S2 (conditioning) and 1m aerial (target).

    All three conditioning sources are loaded from the 5m NPZ directory.
    """

    S1_KEYS    = ["s1_vv", "s1_vh"]
    S2_KEYS    = ["s2_r", "s2_g", "s2_b", "s2_nir"]
    AERIAL_KEYS = ["aerial_r", "aerial_g", "aerial_b", "aerial_nir"]

    def __init__(self, root_5m, root_1m):
        super().__init__()
        files_5m = {p.stem: p for p in sorted(pathlib.Path(root_5m).glob("*.npz"))}
        files_1m = {p.stem: p for p in sorted(pathlib.Path(root_1m).glob("*.npz"))}
        common = sorted(files_5m.keys() & files_1m.keys())
        self.pairs = [(files_5m[k], files_1m[k]) for k in common]

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[Fusion5mTo1mDataset] {len(self.pairs)} paired tiles "
                  f"(5m dir: {len(files_5m)}, 1m dir: {len(files_1m)})")

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _zero_pad(tensor, target_size):
        _, h, w = tensor.shape
        pad_h = target_size - h
        pad_w = target_size - w
        if pad_h == 0 and pad_w == 0:
            return tensor
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        return F.pad(tensor, padding, mode="constant", value=0)

    def __getitem__(self, idx):
        path_5m, path_1m = self.pairs[idx]
        with np.load(path_5m, allow_pickle=True) as d5:
            s1 = torch.from_numpy(
                np.stack([d5[k].astype(np.float32) for k in self.S1_KEYS], axis=0)
            )
            s2 = torch.from_numpy(
                np.stack([d5[k].astype(np.float32) for k in self.S2_KEYS], axis=0)
            )
            aerial_5m = torch.from_numpy(
                np.stack([d5[k].astype(np.float32) for k in self.AERIAL_KEYS], axis=0)
            )
        with np.load(path_1m, allow_pickle=True) as d1:
            aerial = torch.from_numpy(
                np.stack([d1[k].astype(np.float32) for k in self.AERIAL_KEYS], axis=0)
            )

        s1        = self._zero_pad(s1,        LR_PAD_SIZE)
        s2        = self._zero_pad(s2,        LR_PAD_SIZE)
        aerial_5m = self._zero_pad(aerial_5m, AERIAL_5M_PAD_SIZE)
        aerial    = self._zero_pad(aerial,    HR_PAD_SIZE)

        return {"s1": s1, "s2": s2, "aerial_5m": aerial_5m, "aerial": aerial,
                "path": str(path_1m)}


# ──────────────────────────────────────────────────────────────────────────────
# Latent dataset (pre-computed VAE latents)
# ──────────────────────────────────────────────────────────────────────────────

class LatentFusion5mTo1mDataset(Dataset):
    """Loads pre-computed VAE latents for 5m→1m training (S1+S2 variant).

    Expects .pt files containing:
        z_aerial    : (4, 128, 128) float16  — 1m aerial latent (scaled)
        z_aerial_5m : (4, 128, 128) float16  — 5m conditioning latent (unscaled)
        z_s2        : (4, 128, 128) float16  — S2 conditioning latent (unscaled)

    S1 is loaded from the 5m NPZ on-the-fly.
    Note: z_s2 must be added to the precompute script to use this path.
    """

    S1_KEYS = ["s1_vv", "s1_vh"]

    def __init__(self, npz_dir_5m, latent_dir):
        super().__init__()
        self.latent_dir = pathlib.Path(latent_dir)
        paths_5m = sorted(pathlib.Path(npz_dir_5m).glob("*.npz"))
        self.paths = [p for p in paths_5m if (self.latent_dir / f"{p.stem}.pt").exists()]

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[LatentFusion5mTo1mDataset] {len(self.paths)}/{len(paths_5m)} tiles "
                  f"(latent_dir={self.latent_dir})")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        lat = torch.load(self.latent_dir / f"{path.stem}.pt", map_location="cpu", weights_only=True)
        with np.load(path, allow_pickle=True) as f:
            s1 = torch.from_numpy(
                np.stack([f[k].astype(np.float32) for k in self.S1_KEYS])
            )
        s1 = Fusion5mTo1mDataset._zero_pad(s1, LR_PAD_SIZE)
        s1_cond = normalize_s1(s1, stage="norm")
        return {
            "z_aerial":     lat["z_aerial"],
            "z_aerial_5m":  lat["z_aerial_5m"],
            "z_s2":         lat["z_s2"],
            "s1_cond":      s1_cond,
            "path":         str(path),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────────────────────────────────────────

class AugmentedDataset(Dataset):
    """Random flips + 90° rotations applied identically to all spatial inputs."""

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        if "z_aerial" in sample:
            aug_keys = ["z_aerial", "z_aerial_5m", "z_s2", "s1_cond"]
        else:
            aug_keys = ["s1", "s2", "aerial_5m", "aerial"]
        tensors = [sample[k] for k in aug_keys]

        if torch.rand(1).item() > 0.5:
            tensors = [torch.flip(t, dims=[-1]) for t in tensors]
        if torch.rand(1).item() > 0.5:
            tensors = [torch.flip(t, dims=[-2]) for t in tensors]
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            tensors = [torch.rot90(t, k, dims=[-2, -1]) for t in tensors]

        return {**sample, **dict(zip(aug_keys, tensors))}


# ──────────────────────────────────────────────────────────────────────────────
# DataModule
# ──────────────────────────────────────────────────────────────────────────────

class FusionDataModule5To1(pl.LightningDataModule):
    """DataModule for paired 5m+1m NPZ tiles (S1+S2 conditioning variant)."""

    def __init__(self, data_dir_5m, data_dir_1m, batch_size=2, num_workers=4,
                 train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42, augment=False,
                 latent_dir=None):
        super().__init__()
        self.data_dir_5m = data_dir_5m
        self.data_dir_1m = data_dir_1m
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
        self.augment = augment
        self.latent_dir = latent_dir
        self.train_ds = self.val_ds = self.test_ds = None

    def setup(self, stage=None):
        if self.train_ds is None:
            if self.latent_dir:
                full_ds = LatentFusion5mTo1mDataset(self.data_dir_5m, self.latent_dir)
            else:
                full_ds = Fusion5mTo1mDataset(self.data_dir_5m, self.data_dir_1m)
            self.train_ds, self.val_ds, self.test_ds = random_split(
                full_ds,
                [self.train_frac, self.val_frac, self.test_frac],
                generator=torch.Generator().manual_seed(self.seed),
            )
            if self.augment:
                self.train_ds = AugmentedDataset(self.train_ds)
            print(f"[Split] Train: {len(self.train_ds)}, Val: {len(self.val_ds)}, "
                  f"Test: {len(self.test_ds)}")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────

class LitUNetDenoiser(pl.LightningModule):
    """UNet denoiser for 5m→1m super-resolution with S1 + S2 conditioning.

    Conditioning: 5m aerial VAE (4ch) + S1 (2ch) + S2 VAE (4ch) = 10ch total.
    Requires config_5m_to_1m_with_s2.yaml (in_channels: 14).
    """

    def __init__(self, config, vae_ckpt=None, lr=1e-4, max_epochs=100,
                 warmup_epochs=5, cfg_dropout=0.15, no_warmup=False, use_precomputed=False):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.lr = lr
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.cfg_dropout = cfg_dropout
        self.no_warmup = no_warmup
        self.use_precomputed = use_precomputed

        ch_mult = list(config.first_stage_config.ch_mult)
        self.vae_downscale = 2 ** (len(ch_mult) - 1)

        self.ldm = LatentDiffusion(
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
            scale_factor=0.18215,
        )

        if vae_ckpt is not None:
            self._load_vae(vae_ckpt)

        for p in self.ldm.first_stage_model.parameters():
            p.requires_grad = False
        self.ldm.first_stage_model.eval()

        vae_n = sum(p.numel() for p in self.ldm.first_stage_model.parameters())
        unet_n = sum(p.numel() for p in self.ldm.model.parameters())
        unet_train = sum(p.numel() for p in self.ldm.model.parameters() if p.requires_grad)
        print(f"VAE:  {vae_n:,} params (frozen)")
        print(f"UNet: {unet_n:,} params ({unet_train:,} trainable)")

    # ── VAE loading ──────────────────────────────────────────────────────────

    def _load_vae(self, ckpt_path):
        print(f"Loading VAE weights from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        cleaned = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            if k.startswith("disc.") or k.startswith("lpips_fn."):
                continue
            if k.startswith("vae."):
                k = k[4:]
            cleaned[k] = v
        missing, unexpected = self.ldm.first_stage_model.load_state_dict(cleaned, strict=True)
        print(f"  VAE loaded: {len(cleaned)} keys "
              f"(missing: {len(missing)}, unexpected: {len(unexpected)})")

    # ── Encoding helpers ─────────────────────────────────────────────────────

    def _build_conditioning(self, aerial_5m, s1, s2):
        """Build 10-channel conditioning: VAE(5m aerial) [4ch] + S1 [2ch] + VAE(S2) [4ch]."""
        hr_size = HR_PAD_SIZE
        latent_size = hr_size // self.vae_downscale  # 128
        vae_dtype = next(self.ldm.first_stage_model.parameters()).dtype

        aerial_5m_norm = normalize_aerial(aerial_5m, stage="norm")
        aerial_5m_up = F.interpolate(aerial_5m_norm, size=(hr_size, hr_size),
                                     mode="bilinear", align_corners=False)
        s2_norm = normalize_s2(s2, stage="norm")
        s2_up = F.interpolate(s2_norm, size=(hr_size, hr_size),
                              mode="bilinear", align_corners=False)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            cond_5m = self.ldm.first_stage_model.encode(
                aerial_5m_up.to(vae_dtype)).sample()
            cond_s2 = self.ldm.first_stage_model.encode(
                s2_up.to(vae_dtype)).sample()

        s1_norm = normalize_s1(s1, stage="norm")
        cond_s1 = F.interpolate(s1_norm, size=(latent_size, latent_size),
                                mode="bilinear", align_corners=False)

        return torch.cat([cond_5m, cond_s1, cond_s2], dim=1)  # (B, 10, 128, 128)

    def _encode_aerial(self, aerial):
        aerial_norm = normalize_aerial(aerial, stage="norm")
        vae_dtype = next(self.ldm.first_stage_model.parameters()).dtype
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            posterior = self.ldm.encode_first_stage(aerial_norm.to(vae_dtype))
            z_0 = self.ldm.get_first_stage_encoding(posterior)
        return z_0

    # ── Shared step ──────────────────────────────────────────────────────────

    def _shared_step(self, batch):
        if "z_aerial" in batch:
            z_0          = batch["z_aerial"].float().to(self.device)
            cond_5m      = batch["z_aerial_5m"].float().to(self.device)
            s1_cond      = batch["s1_cond"].to(self.device)
            cond_s2      = batch["z_s2"].float().to(self.device)
            conditioning = torch.cat([cond_5m, s1_cond, cond_s2], dim=1)
        else:
            z_0          = self._encode_aerial(batch["aerial"])
            conditioning = self._build_conditioning(
                batch["aerial_5m"], batch["s1"], batch["s2"]
            )

        if self.cfg_dropout > 0 and self.training:
            mask = (torch.rand(conditioning.shape[0], 1, 1, 1,
                               device=conditioning.device) > self.cfg_dropout)
            conditioning = conditioning * mask

        B = z_0.shape[0]
        t = torch.randint(0, self.ldm.num_timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(z_0)
        z_t = self.ldm.q_sample(x_start=z_0, t=t, noise=noise)
        noise_pred = self.ldm.apply_model(z_t, t, cond=conditioning)
        return F.mse_loss(noise_pred, noise)

    # ── Lightning interface ──────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.ldm.model.parameters(), lr=self.lr)
        if self.no_warmup:
            scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        else:
            warmup = LinearLR(optimizer, start_factor=1.0 / self.warmup_epochs,
                              total_iters=self.warmup_epochs)
            cosine = CosineAnnealingLR(optimizer, T_max=self.max_epochs - self.warmup_epochs)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                                     milestones=[self.warmup_epochs])
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def on_fit_start(self):
        if self.use_precomputed:
            self.ldm.first_stage_model.cpu()
            torch.cuda.empty_cache()
            print("[UNet] VAE offloaded to CPU — using pre-computed latents")

    def on_train_epoch_start(self):
        if not self.use_precomputed:
            self.ldm.first_stage_model.eval()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train 5m→1m UNet denoiser (cascade stage 2, S1+S2 conditioning)"
    )
    parser.add_argument("--data_dir_5m", type=str, required=True,
                        help="Path to 5m NPZ tile directory (must contain S1, S2, and 5m aerial)")
    parser.add_argument("--data_dir_1m", type=str, required=True,
                        help="Path to 1m NPZ tile directory")
    parser.add_argument("--vae_ckpt", type=str, default=None,
                        help="Path to pretrained 1m VAE checkpoint")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (default: config_5m_to_1m_with_s2.yaml)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--no_warmup", action="store_true", default=False)
    parser.add_argument("--cfg_dropout", type=float, default=0.15)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (epochs). 0 = disabled.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--latent_dir", type=str, default=None,
                        help="Pre-computed latent directory. "
                             "Latents must include z_s2 — see LatentFusion5mTo1mDataset.")
    parser.add_argument("--find_lr", action="store_true", default=False,
                        help="Run LR range test and save plot, then exit")
    parser.add_argument("--log_dir", type=str, default="lightning_logs")
    args = parser.parse_args()

    if args.vae_ckpt is None:
        print("WARNING: --vae_ckpt not set. VAE weights will be randomly initialized.")

    if args.config is None:
        config_path = (pathlib.Path(__file__).resolve().parent.parent
                       / "opensr_model" / "configs" / "config_5m_to_1m_with_s2.yaml")
    else:
        config_path = pathlib.Path(args.config)
    config = OmegaConf.load(config_path)
    print(f"Config: {config_path}")

    model = LitUNetDenoiser(
        config, vae_ckpt=args.vae_ckpt, lr=args.lr, max_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs, cfg_dropout=args.cfg_dropout,
        no_warmup=args.no_warmup, use_precomputed=args.latent_dir is not None,
    )
    dm = FusionDataModule5To1(
        args.data_dir_5m, args.data_dir_1m,
        batch_size=args.batch_size, num_workers=args.num_workers,
        train_frac=args.train_frac, val_frac=args.val_frac,
        test_frac=args.test_frac, augment=args.augment,
        latent_dir=args.latent_dir,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/5to1m_with_s2/unet",
            filename="unet5to1-s1s2-{epoch:04d}-{val_loss:.6f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if args.patience > 0:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=args.patience,
                                       mode="min", verbose=True))

    logger = TensorBoardLogger(save_dir=args.log_dir, name="unet_5to1m_with_s2")

    num_gpus = torch.cuda.device_count()
    accelerator = "gpu" if num_gpus > 0 else "cpu"
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=num_gpus if accelerator == "gpu" else "auto",
        strategy=DDPStrategy(find_unused_parameters=False) if num_gpus > 1 else "auto",
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )

    if args.find_lr:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, dm, min_lr=1e-7, max_lr=1, num_training=200)
        suggested_lr = lr_finder.suggestion()
        print(f"\nSuggested LR: {suggested_lr:.2e}")
        fig = lr_finder.plot(suggest=True)
        fig.savefig("lr_finder_5to1m_with_s2.png")
        print("LR finder plot saved to: lr_finder_5to1m_with_s2.png")
        return

    trainer.fit(model, dm, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
