"""Phase 1: AutoencoderKL Training on Aerial RGBNIR — Full Paper Loss
=================================================================
Powered by PyTorch Lightning.

This is the FIRST step in the Latent Diffusion SR pipeline.
We train the VAE to compress 256x256 aerial RGBNIR images into
64x64x4 latent representations and reconstruct them faithfully.

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
  NPZ tiles with aerial_r, aerial_g, aerial_b, aerial_nir (uint8, 200x200).
  Zero-padded to 256x256 by FusionDataset.  Normalized to [-1,1].

Usage:
    python train_vae.py --data_dir /path/to/npz_tiles
    python train_vae.py --data_dir /path/to/npz_tiles --epochs 200 --batch_size 8 --precision 16-mixed

After training, use the checkpoint for UNet training (Phase 2):
    python train_unet.py --data_dir /path/to/npz_tiles --vae_ckpt checkpoints/vae/last.ckpt
"""

import argparse
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
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path so opensr_model is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from opensr_model.autoencoder.autoencoder import AutoencoderKL
from opensr_model.data import FusionDataset
from opensr_model.utils import normalize_aerial


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

    dxx = rx.t() + rx - 2.0 * xx
    dyy = ry.t() + ry - 2.0 * yy
    dxy = rx.t() + ry - 2.0 * xy

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
    z_flat = z.reshape(B, -1)       # (B, C*H*W)
    p_flat = z_prior.reshape(B, -1)

    K_zz, K_pp, K_zp = _rbf_kernel(z_flat, p_flat)

    n = B
    mmd = (K_zz.sum() / (n * (n - 1 + 1e-8))
           + K_pp.sum() / (n * (n - 1 + 1e-8))
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
        return self.lpips_fn(reconstruction[:, idx], target[:, idx]).mean()

    def _vae_loss(self, batch):
        """Full VAE loss: WD + MAE + GAN(generator) + LPIPS."""
        aerial = batch["aerial"]  # (B, 4, 256, 256) float32 [0, 255]
        x = normalize_aerial(aerial, stage="norm")  # -> [-1, 1]

        # Forward through VAE
        reconstruction, posterior = self.vae(x, sample_posterior=True)
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
        self.log("train_loss", losses["total"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_wd", losses["wd"], on_step=False, on_epoch=True)
        self.log("train_mae", losses["mae"], on_step=False, on_epoch=True)
        self.log("train_gan_g", losses["gan_g"], on_step=False, on_epoch=True)
        self.log("train_lpips", losses["lpips"], on_step=False, on_epoch=True)
        self.log("train_disc", d_loss, on_step=False, on_epoch=True)

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

    def lr_schedulers(self):
        # Ensure both schedulers are stepped each epoch
        scheds = super().lr_schedulers()
        if not isinstance(scheds, (list, tuple)):
            scheds = [scheds]
        return scheds

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
    random_split strategy (80/10/10 by default, seed=42).
    """

    def __init__(self, data_dir: str, batch_size: int = 4, num_workers: int = 4,
                 train_frac: float = 0.95, val_frac: float = 0.05,
                 test_frac: float = 0.0, seed: int = 42):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
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
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Train VAE (AutoencoderKL) on aerial RGBNIR — full paper loss"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to NPZ tile directory")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (default: config_10m.yaml)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
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
    parser.add_argument("--gan_warmup_epochs", type=int, default=10,
                        help="Epochs before GAN discriminator activates")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_frac", type=float, default=0.95)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--test_frac", type=float, default=0.0)
    parser.add_argument("--precision", type=str, default="32",
                        help="Training precision: 32, 16-mixed, bf16-mixed")
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of GPUs (0 = CPU)")
    parser.add_argument("--log_dir", type=str, default="lightning_logs",
                        help="TensorBoard log directory")
    args = parser.parse_args()

    # ── Config ──────────────────────────────────────────────────────────────
    if args.config is None:
        config_path = (pathlib.Path(__file__).resolve().parent.parent
                       / "opensr_model" / "configs" / "config_10m.yaml")
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
        test_frac=args.test_frac,
    )

    # ── Callbacks ───────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/vae",
            filename="vae-{epoch:04d}-{val_loss:.6f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ── Logger ──────────────────────────────────────────────────────────────
    logger = TensorBoardLogger(save_dir=args.log_dir, name="vae_aerial")

    # ── Trainer ─────────────────────────────────────────────────────────────
    accelerator = "gpu" if args.devices > 0 and torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=args.devices if accelerator == "gpu" else "auto",
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
    )

    # ── Train ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Phase 1: Training AutoencoderKL on aerial RGBNIR")
    print(f"  Input:   (B, 4, 256, 256)  aerial RGBNIR")
    print(f"  Latent:  (B, 4, 64, 64)")
    print(f"  Loss:    lam_WD={args.lam_wd} * WD(z)")
    print(f"         + lam_MAE={args.lam_mae} * MAE(xhat)")
    print(f"         + lam_GAN={args.lam_gan} * GAN(xhat)  [warm-up: {args.gan_warmup_epochs} epochs]")
    print(f"         + lam_LPIPS={args.lam_lpips} * LPIPS(xhat)")
    print(f"{'=' * 60}\n")

    trainer.fit(model, dm)

    print(f"\nPhase 1 complete!")
    print(f"  Best checkpoint: checkpoints/vae/")
    print(f"\nNext step — Phase 2: Train UNet denoiser")
    print(f"  python train_unet.py --data_dir {args.data_dir} "
          f"--vae_ckpt checkpoints/vae/last.ckpt")


if __name__ == "__main__":
    main()
