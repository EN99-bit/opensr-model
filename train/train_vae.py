"""Phase 1: AutoencoderKL Training on Aerial RGBNIR — Full Paper Loss
=================================================================
Powered by PyTorch Lightning.

This is the FIRST step in the Latent Diffusion SR pipeline.
We train the VAE to compress 1024x1024 aerial RGBNIR images into
128x128x4 latent representations and reconstruct them faithfully.

TRAINING OBJECTIVE (following LDSR-S2 paper, Eq. 1):
  L_total = λ_WD  · L_WD(z)
          + λ_MAE · L_MAE(x̂)
          + λ_GAN · L_GAN(x̂)
          + λ_LPIPS · L_LPIPS(x̂)

  - WD:    Wasserstein Distance (MMD approximation) — regularizes the latent
           space distribution toward N(0,I). Replaces KL for more stable
           training and better-structured latent space.
  - MAE:   L1 pixel reconstruction loss.
  - GAN:   PatchGAN discriminator with hinge loss — forces realistic
           reconstructions. Activated after a warm-up period.
  - LPIPS: Learned Perceptual Image Patch Similarity (VGG backbone) —
           perceptual quality. Random 3-of-4 bands selected each step
           (VGG expects 3ch input, paper Section IV-A).

DATA:
  NPZ tiles with aerial_r, aerial_g, aerial_b, aerial_nir (uint8, 1000x1000).
  Zero-padded to 1024x1024 by FusionDataset.  Normalized to [-1,1].

Usage:
    python train_vae.py --data_dir /path/to/npz_tiles
    python train_vae.py --data_dir /path/to/npz_tiles --epochs 200 --batch_size 2 --precision 16-mixed

After training, use the checkpoint for UNet training (Phase 2):
    python train_unet.py --data_dir /path/to/npz_tiles --vae_ckpt checkpoints/1m/vae/last.ckpt
"""

import argparse
import os
import pathlib
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path so opensr_model is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from opensr_model.autoencoder.autoencoder import AutoencoderKL
from opensr_model.data import FusionDataset
from opensr_model.utils import normalize_aerial


# ──────────────────────────────────────────────────────────────────────────────
# D4 augmentation wrapper (8 deterministic orientations per tile)
# ──────────────────────────────────────────────────────────────────────────────

# The 8 elements of the dihedral group D4: (k rotations of 90°, flip)
_D4 = [(k, flip) for k in range(4) for flip in (False, True)]


def _apply_d4(tensor: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    if k:
        tensor = torch.rot90(tensor, k, dims=[-2, -1])
    if flip:
        tensor = torch.flip(tensor, dims=[-1])
    return tensor


class D4Dataset(torch.utils.data.Dataset):
    """Expands a dataset 8× by returning all D4 orientations of each tile.

    Index mapping: sample i → base tile i//8, transform i%8.
    All spatial tensors (s1, s2, aerial) get the same transform so they stay aligned.
    """

    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds) * 8

    def __getitem__(self, idx):
        sample = self.base_ds[idx // 8]
        k, flip = _D4[idx % 8]
        if k or flip:
            sample = {**sample,
                      "s1":     _apply_d4(sample["s1"],     k, flip),
                      "s2":     _apply_d4(sample["s2"],     k, flip),
                      "aerial": _apply_d4(sample["aerial"], k, flip)}
        return sample


# ──────────────────────────────────────────────────────────────────────────────
# Random crop wrapper
# ──────────────────────────────────────────────────────────────────────────────

class CroppedDataset(torch.utils.data.Dataset):
    """Randomly crops aerial (and S1/S2 at matching location) to crop_size×crop_size.

    S1/S2 are at a lower resolution than aerial (LR_PAD_SIZE vs HR_PAD_SIZE).
    The crop location is scaled proportionally so all tensors stay geographically aligned.
    crop_size must be divisible by the VAE downscale factor (e.g. 8 for ch_mult=[1,2,4,8]).
    """

    def __init__(self, base_ds, crop_size: int):
        self.base_ds = base_ds
        self.crop_size = crop_size

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = self.base_ds[idx]
        _, H, W = sample["aerial"].shape
        top  = torch.randint(0, H - self.crop_size + 1, (1,)).item()
        left = torch.randint(0, W - self.crop_size + 1, (1,)).item()

        aerial = sample["aerial"][:, top:top + self.crop_size, left:left + self.crop_size]

        # Scale crop window to LR resolution and crop S1/S2 consistently
        lr_h = sample["s1"].shape[-2]
        scale = lr_h / H
        lr_top  = round(top  * scale)
        lr_left = round(left * scale)
        lr_crop = round(self.crop_size * scale)
        s1 = sample["s1"][:, lr_top:lr_top + lr_crop, lr_left:lr_left + lr_crop]
        s2 = sample["s2"][:, lr_top:lr_top + lr_crop, lr_left:lr_left + lr_crop]

        return {**sample, "aerial": aerial, "s1": s1, "s2": s2}


# ──────────────────────────────────────────────────────────────────────────────
# Wasserstein Distance (MMD approximation)
# ──────────────────────────────────────────────────────────────────────────────

def _rbf_kernel(x, y, bandwidths=(0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)):
    """Compute RBF (Gaussian) kernel matrix between two sets of samples.

    Args:
        x: (N, D) tensor of samples.
        y: (M, D) tensor of samples.
        bandwidths: Tuple of kernel bandwidths (sigma^2 values).

    Returns:
        Scalar kernel value averaged over all bandwidths.
    """
    xx = x @ x.t()
    yy = y @ y.t()
    xy = x @ y.t()
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = (rx.t() + rx - 2.0 * xx).clamp(min=0)
    dyy = (ry.t() + ry - 2.0 * yy).clamp(min=0)
    dxy = (rx.t() + ry - 2.0 * xy).clamp(min=0)

    K_xx, K_yy, K_xy = (
        torch.zeros_like(xx),
        torch.zeros_like(yy),
        torch.zeros_like(xy),
    )
    for bw in bandwidths:
        K_xx = K_xx + torch.exp(-dxx / (2.0 * bw))
        K_yy = K_yy + torch.exp(-dyy / (2.0 * bw))
        K_xy = K_xy + torch.exp(-dxy / (2.0 * bw))

    return K_xx, K_yy, K_xy


def mmd_loss(z, z_prior):
    """Maximum Mean Discrepancy between encoded z and prior N(0,I).

    Used as Wasserstein Distance approximation for WAE regularization.

    Args:
        z:       (B, C, H, W) — encoded latent samples.
        z_prior: (B, C, H, W) — samples from N(0, I) with same shape.

    Returns:
        Scalar MMD loss.
    """
    B = z.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)

    # nan_to_num handles fp16 NaN/Inf before kernel computation (clamp alone does not fix NaN)
    z_flat = z.float().nan_to_num(0.0, posinf=1e4, neginf=-1e4).reshape(B, -1)
    p_flat = z_prior.float().reshape(B, -1)

    # Disable autocast: matmul is autocast-eligible, so fp32 z_flat would be
    # cast to fp16, causing dot products (4096 × (1e4)²) to overflow → NaN.
    with torch.autocast(device_type="cuda", enabled=False):
        K_zz, K_pp, K_zp = _rbf_kernel(z_flat, p_flat)

    n = B
    mmd = (K_zz.sum() / (n * (n - 1))
           + K_pp.sum() / (n * (n - 1))
           - 2.0 * K_zp.sum() / (n * n))
    return mmd


# ──────────────────────────────────────────────────────────────────────────────
# PatchGAN Discriminator
# ──────────────────────────────────────────────────────────────────────────────

class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for 4-channel (RGBNIR) images.

    Outputs a spatial map of real/fake predictions (no sigmoid — use with
    hinge loss).  Receptive field covers ~70x70 patches.

    Args:
        in_channels: Number of input channels (4 for RGBNIR).
        ndf:         Base number of discriminator filters.
        n_layers:    Number of downsampling conv layers.
    """

    def __init__(self, in_channels: int = 4, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch_prev = ndf
        for i in range(1, n_layers):
            ch_next = min(ch_prev * 2, ndf * 8)
            layers += [
                nn.Conv2d(ch_prev, ch_next, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch_next),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch_prev = ch_next
        # Final layer — stride 1
        ch_next = min(ch_prev * 2, ndf * 8)
        layers += [
            nn.Conv2d(ch_prev, ch_next, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_next),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        # Output — 1 channel prediction map
        layers += [
            nn.Conv2d(ch_next, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def hinge_loss_d(real_logits, fake_logits):
    """Discriminator hinge loss."""
    return 0.5 * (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean())


def hinge_loss_g(fake_logits):
    """Generator hinge loss (VAE wants discriminator to think reconstructions are real)."""
    return -fake_logits.mean()


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────

class LitVAE(pl.LightningModule):
    """PyTorch Lightning module for training AutoencoderKL on aerial RGBNIR.

    Implements the full LDSR-S2 paper loss (Eq. 1):
      L = λ_WD · MMD(z) + λ_MAE · MAE(x̂) + λ_GAN · GAN(x̂) + λ_LPIPS · LPIPS(x̂)

    Uses manual optimization for the dual-optimizer GAN setup.
    """

    def __init__(self, config, lr: float = 1e-4, lr_disc: float = 4e-4,
                 max_epochs: int = 100,
                 lam_wd: float = 1.0, lam_mae: float = 1.0,
                 lam_gan: float = 0.5, lam_lpips: float = 1.0,
                 gan_warmup_epochs: int = 10):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.automatic_optimization = False  # Manual optimization for GAN
        self.config = config
        self.lr = lr
        self.lr_disc = lr_disc
        self.max_epochs = max_epochs
        self.lam_wd = lam_wd
        self.lam_mae = lam_mae
        self.lam_gan = lam_gan
        self.lam_lpips = lam_lpips
        self.gan_warmup_epochs = gan_warmup_epochs

        # Build AutoencoderKL from config
        ddconfig = dict(config.first_stage_config)
        embed_dim = ddconfig.pop("embed_dim")
        self.vae = AutoencoderKL(ddconfig, embed_dim)

        # PatchGAN discriminator (4-channel RGBNIR input)
        self.disc = PatchDiscriminator(in_channels=4, ndf=64, n_layers=3)

        # LPIPS perceptual loss (VGG backbone, frozen)
        self.lpips_fn = lpips.LPIPS(net="vgg")
        self.lpips_fn.eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad = False

        # Log param counts
        vae_n = sum(p.numel() for p in self.vae.parameters())
        disc_n = sum(p.numel() for p in self.disc.parameters())
        print(f"VAE:           {vae_n:,} params")
        print(f"Discriminator: {disc_n:,} params")

    @property
    def _gan_active(self) -> bool:
        """GAN loss is activated only after warm-up period."""
        return self.current_epoch >= self.gan_warmup_epochs

    # ── Loss computation ─────────────────────────────────────────────────────

    def _compute_lpips(self, reconstruction, target):
        """LPIPS on random 3-of-4 bands (VGG expects 3ch input).

        Paper Section IV-A: randomly select three of the 4 bands at each
        training step to calculate LPIPS.
        """
        idx = sorted(random.sample(range(4), 3))
        # Disable autocast: Conv2d is autocast-eligible, so VGG runs in fp16 even
        # with fp32 inputs. With reconstruction values up to ±10, VGG activations
        # can overflow fp16 → NaN. Force fp32 for the entire VGG forward pass.
        with torch.autocast(device_type="cuda", enabled=False):
            return self.lpips_fn(reconstruction[:, idx].float(), target[:, idx].float()).mean()

    def _vae_loss(self, batch):
        """Full VAE loss: WD + MAE + GAN(generator) + LPIPS."""
        aerial = batch["aerial"]  # (B, 4, H, W) float32 [0, 255]
        x = normalize_aerial(aerial, stage="norm")  # -> [-1, 1]

        # Forward through VAE
        reconstruction, posterior = self.vae(x, sample_posterior=True)
        reconstruction = reconstruction.float().nan_to_num(0.0).clamp(-10, 10).to(x.dtype)
        z = posterior.sample()

        # ── Wasserstein Distance (MMD) ───────────────────────────────────
        z_prior = torch.randn_like(z)
        wd_loss = mmd_loss(z, z_prior)

        # ── MAE reconstruction loss ──────────────────────────────────────
        mae_loss = F.l1_loss(reconstruction, x)

        # ── LPIPS perceptual loss ────────────────────────────────────────
        # VGG weights are frozen, but gradients must flow through reconstruction
        lpips_loss = self._compute_lpips(reconstruction, x)

        # ── GAN generator loss (after warm-up) ──────────────────────────
        if self._gan_active:
            fake_logits = self.disc(reconstruction)
            gan_g_loss = hinge_loss_g(fake_logits)
        else:
            gan_g_loss = torch.tensor(0.0, device=self.device)

        # ── Total generator/VAE loss ─────────────────────────────────────
        total = (self.lam_wd * wd_loss
                 + self.lam_mae * mae_loss
                 + self.lam_gan * gan_g_loss
                 + self.lam_lpips * lpips_loss)

        losses = {
            "total": total,
            "wd": wd_loss,
            "mae": mae_loss,
            "gan_g": gan_g_loss,
            "lpips": lpips_loss,
        }
        if not torch.isfinite(total) and self.global_rank == 0:
            print(f"[NaN] wd={wd_loss.item():.4f} mae={mae_loss.item():.4f} "
                  f"lpips={lpips_loss.item():.4f} "
                  f"recon_max={reconstruction.abs().max().item():.2f} "
                  f"x_max={x.abs().max().item():.2f}")
        return losses, reconstruction, x

    def _disc_loss(self, reconstruction, target):
        """Discriminator hinge loss on real vs. fake (reconstructed) images."""
        real_logits = self.disc(target)
        fake_logits = self.disc(reconstruction.detach())
        return hinge_loss_d(real_logits, fake_logits)

    # ── Lightning interface ──────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        opt_vae, opt_disc = self.optimizers()

        # ── Step 1: Train VAE (generator) ────────────────────────────────
        losses, reconstruction, target = self._vae_loss(batch)
        opt_vae.zero_grad()
        self.manual_backward(losses["total"])
        self.clip_gradients(opt_vae, gradient_clip_val=1.0)
        opt_vae.step()

        # ── Step 2: Train Discriminator (after warm-up) ──────────────────
        if self._gan_active:
            d_loss = self._disc_loss(reconstruction, target)
            opt_disc.zero_grad()
            self.manual_backward(d_loss)
            self.clip_gradients(opt_disc, gradient_clip_val=1.0)
            opt_disc.step()
        else:
            d_loss = torch.tensor(0.0, device=self.device)

        # ── Logging ──────────────────────────────────────────────────────
        self.log("train_loss", losses["total"], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_wd", losses["wd"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_mae", losses["mae"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_gan_g", losses["gan_g"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_lpips", losses["lpips"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_disc", d_loss, on_step=False, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        losses, _, _ = self._vae_loss(batch)
        self.log("val_loss", losses["total"], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_wd", losses["wd"], on_epoch=True, sync_dist=True)
        self.log("val_mae", losses["mae"], on_epoch=True, sync_dist=True)
        self.log("val_gan_g", losses["gan_g"], on_epoch=True, sync_dist=True)
        self.log("val_lpips", losses["lpips"], on_epoch=True, sync_dist=True)
        return losses["total"]

    def configure_optimizers(self):
        opt_vae = torch.optim.AdamW(self.vae.parameters(), lr=self.lr)
        opt_disc = torch.optim.AdamW(self.disc.parameters(), lr=self.lr_disc)
        sch_vae = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_vae, T_max=self.max_epochs
        )
        sch_disc = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_disc, T_max=self.max_epochs
        )
        return (
            {"optimizer": opt_vae, "lr_scheduler": {"scheduler": sch_vae, "interval": "epoch"}},
            {"optimizer": opt_disc, "lr_scheduler": {"scheduler": sch_disc, "interval": "epoch"}},
        )

    def on_train_epoch_end(self):
        # Step LR schedulers manually (required with manual optimization)
        scheds = self.lr_schedulers()
        for sch in scheds:
            if sch is not None:
                sch.step()


# ──────────────────────────────────────────────────────────────────────────────
# LightningDataModule
# ──────────────────────────────────────────────────────────────────────────────

class AerialDataModule(pl.LightningDataModule):
    """DataModule that loads NPZ tiles for VAE training (only aerial bands used).

    Uses the same FusionDataset as train_unet.py and the same deterministic
    random_split strategy (95/5 by default, seed=42).
    """

    def __init__(self, data_dir: str, batch_size: int = 4, num_workers: int = 4,
                 train_frac: float = 0.95, val_frac: float = 0.05,
                 test_frac: float = 0.0, seed: int = 42, augment: bool = False,
                 crop_size: "int | None" = None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
        self.augment = augment
        self.crop_size = crop_size
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage=None):
        if self.train_ds is None:
            full_ds = FusionDataset(root=self.data_dir, require_aerial=True)
            if self.test_frac > 0:
                self.train_ds, self.val_ds, self.test_ds = random_split(
                    full_ds,
                    [self.train_frac, self.val_frac, self.test_frac],
                    generator=torch.Generator().manual_seed(self.seed),
                )
            else:
                self.train_ds, self.val_ds = random_split(
                    full_ds,
                    [self.train_frac, self.val_frac],
                    generator=torch.Generator().manual_seed(self.seed),
                )
                self.test_ds = []
            if self.crop_size:
                self.train_ds = CroppedDataset(self.train_ds, self.crop_size)
                self.val_ds = CroppedDataset(self.val_ds, self.crop_size)
            if self.augment:
                self.train_ds = D4Dataset(self.train_ds)
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(f"[Split] Train: {len(self.train_ds)}, "
                      f"Val: {len(self.val_ds)}, Test: {len(self.test_ds)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Reconstruction visualisation callback
# ──────────────────────────────────────────────────────────────────────────────

class ReconVizCallback(pl.Callback):
    """Runs ae-test.py after each epoch and saves output as 1m-ae_check-mmd-{epoch:04d}.png."""

    def __init__(self, script_path: str):
        self.script_path = pathlib.Path(script_path).resolve()

    def on_train_epoch_end(self, trainer, _pl_module):
        if trainer.global_rank != 0:
            return
        import subprocess, shutil
        epoch = trainer.current_epoch
        result = subprocess.run(
            [sys.executable, str(self.script_path)],
            cwd=str(self.script_path.parent),
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[ReconViz] ae-test.py failed (epoch {epoch}):\n{result.stderr}")
            return
        src = self.script_path.parent / "1m-ae_check.png"
        dst = self.script_path.parent / f"1m-ae_check-{epoch:04d}.png"
        if src.exists():
            shutil.move(str(src), str(dst))
        else:
            print(f"[ReconViz] ae-test.py succeeded but {src} not found")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Train VAE (AutoencoderKL) on aerial RGBNIR — full paper loss"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to NPZ tile directory")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (default: config_1m.yaml)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="VAE learning rate")
    parser.add_argument("--lr_disc", type=float, default=4e-4,
                        help="Discriminator learning rate")
    # Loss weights (paper Eq. 1)
    parser.add_argument("--lam_wd", type=float, default=1.0,
                        help="Wasserstein distance (MMD) weight")
    parser.add_argument("--lam_mae", type=float, default=1.0,
                        help="MAE reconstruction weight")
    parser.add_argument("--lam_gan", type=float, default=0.5,
                        help="GAN loss weight")
    parser.add_argument("--lam_lpips", type=float, default=1.0,
                        help="LPIPS perceptual loss weight")
    parser.add_argument("--gan_warmup_epochs", type=int, default=1,
                        help="Epochs before GAN discriminator activates")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_frac", type=float, default=0.95)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--test_frac", type=float, default=0.0)
    parser.add_argument("--precision", type=str, default="32",
                        help="Training precision: 32, 16-mixed, bf16-mixed")
    parser.add_argument("--devices", type=int, default=4,
                        help="Number of GPUs (0 = CPU)")
    parser.add_argument("--augment", action="store_true",
                        help="Expand training set 8× with D4 flips+rotations (val unaffected)")
    parser.add_argument("--crop_size", type=int, default=None,
                        help="Random crop size for training (e.g. 256 or 512). "
                             "Must be divisible by VAE downscale factor.")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (epochs). 0 = disabled.")
    parser.add_argument("--log_dir", type=str, default="lightning_logs",
                        help="TensorBoard log directory")
    parser.add_argument("--recon_viz", action="store_true",
                        help="Run ae-test.py after each epoch and save reconstruction images.")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Resume training from checkpoint (restores weights, optimizer, epoch).")
    args = parser.parse_args()

    # ── Config ──────────────────────────────────────────────────────────────
    if args.config is None:
        config_path = (pathlib.Path(__file__).resolve().parent.parent
                       / "opensr_model" / "configs" / "config_1m.yaml")
    else:
        config_path = pathlib.Path(args.config)
    config = OmegaConf.load(config_path)
    print(f"Config: {config_path}")

    # ── Model + Data ────────────────────────────────────────────────────────
    model = LitVAE(
        config, lr=args.lr, lr_disc=args.lr_disc, max_epochs=args.epochs,
        lam_wd=args.lam_wd, lam_mae=args.lam_mae,
        lam_gan=args.lam_gan, lam_lpips=args.lam_lpips,
        gan_warmup_epochs=args.gan_warmup_epochs,
    )
    dm = AerialDataModule(
        args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_frac=args.train_frac, val_frac=args.val_frac,
        test_frac=args.test_frac, augment=args.augment, crop_size=args.crop_size,
    )

    # ── Callbacks ───────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/1m/vae",
            filename="vae-{epoch:04d}-{val_loss:.6f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if args.patience > 0:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=args.patience, mode="min", verbose=True))
    if args.recon_viz:
        ae_script = pathlib.Path(__file__).resolve().parent.parent / "ae-test.py"
        if ae_script.exists():
            callbacks.append(ReconVizCallback(str(ae_script)))

    # ── Logger ──────────────────────────────────────────────────────────────
    logger = TensorBoardLogger(save_dir=args.log_dir, name="vae_aerial")

    # ── Trainer ─────────────────────────────────────────────────────────────
    accelerator = "gpu" if args.devices > 0 and torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=args.devices if accelerator == "gpu" else "auto",
        strategy=DDPStrategy(find_unused_parameters=True) if (accelerator == "gpu" and args.devices > 1) else "auto",
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
    )

    # ── Train ───────────────────────────────────────────────────────────────
    if trainer.global_rank == 0:
        _ch_mult = list(config.first_stage_config.ch_mult)
        _resolution = config.first_stage_config.resolution
        _latent = _resolution // (2 ** (len(_ch_mult) - 1))
        print(f"\n{'=' * 60}")
        print(f"Phase 1: Training AutoencoderKL on aerial RGBNIR")
        print(f"  Input:   (B, 4, {_resolution}, {_resolution})  aerial RGBNIR")
        print(f"  Latent:  (B, 4, {_latent}, {_latent})")
        print(f"  Loss:    lam_WD={args.lam_wd} * WD(z)")
        print(f"         + lam_MAE={args.lam_mae} * MAE(xhat)")
        print(f"         + lam_GAN={args.lam_gan} * GAN(xhat)  [warm-up: {args.gan_warmup_epochs} epochs]")
        print(f"         + lam_LPIPS={args.lam_lpips} * LPIPS(xhat)")
        print(f"{'=' * 60}\n")

    trainer.fit(model, dm, ckpt_path=args.ckpt_path)

    if trainer.global_rank == 0:
        print(f"\nPhase 1 complete!")
        print(f"  Best checkpoint: checkpoints/1m/vae/")
        print(f"\nNext step — Phase 2: Train UNet denoiser")
        print(f"  python train_unet.py --data_dir {args.data_dir} "
              f"--vae_ckpt checkpoints/1m/vae/last.ckpt")


if __name__ == "__main__":
    main()
