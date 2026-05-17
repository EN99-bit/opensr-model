"""
Training script for the UNet denoiser with S1+S2 fusion conditioning.
Powered by PyTorch Lightning.

Usage:
    python train_unet.py --data_dir /path/to/npz_tiles --vae_ckpt /path/to/vae.ckpt
    python train_unet.py --data_dir data --vae_ckpt checkpoints/vae.ckpt --epochs 50 --batch_size 4

Training flow per step (1m aerial, scale_factor=8, padded 128→1024):
    1. aerial (B,4,1024,1024) → normalize [-1,1] → VAE encode → z_0 (B,4,128,128)
    2. S2 (B,4,128,128) → normalize → upsample 1024 → VAE encode → cond_s2 (B,4,128,128)
       S1 (B,2,128,128) → normalize                             → cond_s1 (B,2,128,128)
       conditioning = concat(cond_s2, cond_s1) → (B,6,128,128)
    3. Sample t ~ Uniform(0, T), eps ~ N(0, I)
    4. z_t = sqrt(α̅_t) · z_0 + sqrt(1−α̅_t) · eps
    5. eps_pred = UNet(concat(z_t, conditioning), t)
    6. loss = MSE(eps_pred, eps)
"""

import argparse
import math
import pathlib
import sys

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner import Tuner

# Add project root to path so opensr_model is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from opensr_model.data import FusionDataset, LatentFusionDataset
from opensr_model.diffusion.latentdiffusion import LatentDiffusion
from opensr_model.utils import normalize_s1, normalize_s2, normalize_aerial


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────

class LitUNetDenoiser(pl.LightningModule):
    """PyTorch Lightning module for training the UNet denoiser.

    Wraps a LatentDiffusion model with frozen VAE and trainable UNet.
    Conditioning is built on-the-fly from S1+S2 inputs.
    """

    def __init__(self, config, vae_ckpt: str = None, lr: float = 1e-4, max_epochs: int = 100, warmup_epochs: int = 5, cfg_dropout: float = 0.15, no_warmup: bool = False, use_precomputed: bool = False):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.lr = lr
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.cfg_dropout = cfg_dropout
        self.no_warmup = no_warmup
        self.use_precomputed = use_precomputed

        # Derived sizes from config
        self.scale_factor = config.scale_factor
        ch_mult = list(config.first_stage_config.ch_mult)
        self.vae_downscale = 2 ** (len(ch_mult) - 1)

        # Build LatentDiffusion (contains UNet + VAE)
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
            scale_factor=0.18215
        )

        # Load pretrained VAE weights
        if vae_ckpt is not None:
            self._load_vae(vae_ckpt)

        # Freeze VAE — only UNet trains
        for p in self.ldm.first_stage_model.parameters():
            p.requires_grad = False
        self.ldm.first_stage_model.eval()

        # Log param counts
        vae_n = sum(p.numel() for p in self.ldm.first_stage_model.parameters())
        unet_n = sum(p.numel() for p in self.ldm.model.parameters())
        unet_train = sum(p.numel() for p in self.ldm.model.parameters() if p.requires_grad)
        print(f"VAE:  {vae_n:,} params (frozen)")
        print(f"UNet: {unet_n:,} params ({unet_train:,} trainable)")

    # ── VAE loading ──────────────────────────────────────────────────────────

    def _load_vae(self, ckpt_path: str):
        """Load pretrained VAE weights into ldm.first_stage_model."""
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
        print(f"  VAE loaded: {len(cleaned)} keys (missing: {len(missing)}, unexpected: {len(unexpected)})")

    # ── Encoding helpers ─────────────────────────────────────────────────────

    def _build_conditioning(self, s2, s1):
        """Build 6-channel conditioning: VAE(S2) [4ch] + S1 [2ch] in latent space."""
        lr_size = s2.shape[-1]
        hr_size = lr_size * self.scale_factor
        latent_size = hr_size // self.vae_downscale

        # S2 → normalize → upsample to HR → VAE encode → 4ch latent
        s2_norm = normalize_s2(s2, stage="norm")
        s2_up = F.interpolate(s2_norm, size=(hr_size, hr_size), mode="bilinear", align_corners=False)
        vae_dtype = next(self.ldm.first_stage_model.parameters()).dtype
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            cond_s2 = self.ldm.first_stage_model.encode(s2_up.to(vae_dtype)).sample()

        # S1 → normalize → upsample to latent size → 2ch
        s1_norm = normalize_s1(s1, stage="norm")
        cond_s1 = F.interpolate(s1_norm, size=(latent_size, latent_size), mode="bilinear", align_corners=False)

        return torch.cat([cond_s2, cond_s1], dim=1)

    def _encode_aerial(self, aerial):
        """Encode aerial HR image → VAE latent z_0."""
        aerial_norm = normalize_aerial(aerial, stage="norm")
        vae_dtype = next(self.ldm.first_stage_model.parameters()).dtype
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            posterior = self.ldm.encode_first_stage(aerial_norm.to(vae_dtype))
            z_0 = self.ldm.get_first_stage_encoding(posterior)
        return z_0

    # ── Shared step ──────────────────────────────────────────────────────────

    def _shared_step(self, batch):
        """Diffusion training step shared between train and val.

        Returns MSE loss between predicted and actual noise.
        """
        if "z_aerial" in batch:
            # Pre-computed path — no VAE VRAM
            z_0     = batch["z_aerial"].float().to(self.device)  # (B, 4, 128, 128)
            cond_s2 = batch["z_s2"].float().to(self.device)      # (B, 4, 128, 128)
            cond_s1 = batch["s1_cond"].to(self.device)           # (B, 2, 128, 128)
            conditioning = torch.cat([cond_s2, cond_s1], dim=1)  # (B, 6, 128, 128)
            if self.global_step == 0 and self.training:
                print(f"[z_0] mean={z_0.mean():.4f}  std={z_0.std():.4f}  "
                      f"(expect ~0 and ~1 if latents are well-scaled)")
        else:
            # On-the-fly path — original behaviour
            z_0          = self._encode_aerial(batch["aerial"])
            conditioning = self._build_conditioning(batch["s2"], batch["s1"])

        # Drop conditioning with cfg_dropout probability (classifier-free guidance)
        if self.cfg_dropout > 0 and self.training:
            mask = (torch.rand(conditioning.shape[0], 1, 1, 1, device=conditioning.device) > self.cfg_dropout)
            conditioning = conditioning * mask

        # 3. Sample timestep and noise
        B = z_0.shape[0]
        t = torch.randint(0, self.ldm.num_timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(z_0)

        # 4. Create noisy latent
        z_t = self.ldm.q_sample(x_start=z_0, t=t, noise=noise)

        # 5. UNet predicts noise
        noise_pred = self.ldm.apply_model(z_t, t, cond=conditioning)

        # 6. MSE loss
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
            warmup = LinearLR(optimizer, start_factor=1.0 / self.warmup_epochs, total_iters=self.warmup_epochs)
            cosine = CosineAnnealingLR(optimizer, T_max=self.max_epochs - self.warmup_epochs)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def on_fit_start(self):
        if self.use_precomputed:
            self.ldm.first_stage_model.cpu()
            torch.cuda.empty_cache()
            print("[UNet] VAE offloaded to CPU — using pre-computed latents")

    def on_train_epoch_start(self):
        # Ensure VAE stays frozen and in eval mode (or on CPU if precomputed)
        if not self.use_precomputed:
            self.ldm.first_stage_model.eval()


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation wrapper
# ──────────────────────────────────────────────────────────────────────────────

class AugmentedDataset(Dataset):
    """Wraps a FusionDataset split and applies random flips + 90° rotations.

    The same transform is applied identically to s1, s2, and aerial so that
    spatial alignment is preserved. Only used for the training split.
    """

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        if "z_aerial" in sample:
            aug_keys = ["z_aerial", "z_s2", "s1_cond"]
        else:
            aug_keys = ["s1", "s2", "aerial"]

        tensors = [sample[k] for k in aug_keys]
        if torch.rand(1).item() > 0.5:
            tensors = [torch.flip(t, dims=[-1]) for t in tensors]
        if torch.rand(1).item() > 0.5:
            tensors = [torch.flip(t, dims=[-2]) for t in tensors]
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            tensors = [torch.rot90(t, k, dims=[-2, -1]) for t in tensors]

        return {**sample, **dict(zip(aug_keys, tensors))}


class AugmentedDatasetX8(Dataset):
    """Expands each tile to all 8 elements of the D4 symmetry group (4 rotations × {id, hflip}).

    len = 8 × len(base). Works for both precomputed-latent and raw batches.
    """
    _D4 = [(False,0),(False,1),(False,2),(False,3),(True,0),(True,1),(True,2),(True,3)]

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return 8 * len(self.base)

    def __getitem__(self, idx):
        hflip, nrot = self._D4[idx % 8]
        sample = self.base[idx // 8]
        keys = ["z_aerial","z_s2","s1_cond"] if "z_aerial" in sample else ["s1","s2","aerial"]
        ts = [sample[k] for k in keys]
        if hflip:
            ts = [torch.flip(t, dims=[-1]) for t in ts]
        if nrot:
            ts = [torch.rot90(t, nrot, dims=[-2,-1]) for t in ts]
        return {**sample, **dict(zip(keys, ts))}


# ──────────────────────────────────────────────────────────────────────────────
# LightningDataModule
# ──────────────────────────────────────────────────────────────────────────────

class FusionDataModule(pl.LightningDataModule):
    """DataModule for S1+S2+Aerial NPZ tiles.

    Splits a single directory of NPZ files into train/val/test
    using torch.utils.data.random_split (deterministic seed).
    """

    def __init__(self, data_dir: str, batch_size: int = 2, num_workers: int = 4,
                 train_frac: float = 0.8, val_frac: float = 0.1, test_frac: float = 0.1,
                 seed: int = 42, augment: bool = False, augment_d4: bool = False, latent_dir=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
        self.augment = augment
        self.augment_d4 = augment_d4
        self.latent_dir = latent_dir
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage=None):
        if self.train_ds is None:
            if self.latent_dir:
                full_ds = LatentFusionDataset(root=self.data_dir, latent_dir=self.latent_dir)
            else:
                full_ds = FusionDataset(root=self.data_dir)
            self.train_ds, self.val_ds, self.test_ds = random_split(
                full_ds,
                [self.train_frac, self.val_frac, self.test_frac],
                generator=torch.Generator().manual_seed(self.seed),
            )
            if self.augment_d4:
                self.train_ds = AugmentedDatasetX8(self.train_ds)
            elif self.augment:
                self.train_ds = AugmentedDataset(self.train_ds)
            print(f"[Split] Train: {len(self.train_ds)}, Val: {len(self.val_ds)}, Test: {len(self.test_ds)}")

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
    parser = argparse.ArgumentParser(description="Train UNet denoiser with S1+S2 fusion (Lightning)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to NPZ tile directory")
    parser.add_argument("--vae_ckpt", type=str, default=None, help="Path to pretrained VAE checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config (default: config_10m.yaml)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Epochs to linearly ramp LR from 0 to --lr")
    parser.add_argument("--no_warmup", action="store_true", default=False, help="Disable LR warmup, use plain CosineAnnealingLR")
    parser.add_argument("--cfg_dropout", type=float, default=0.15, help="Probability of dropping conditioning during training for CFG (0 = disabled)")
    parser.add_argument("--augment", action="store_true", default=False, help="Enable random flip+rotation augmentation on training set")
    parser.add_argument("--augment_d4", action="store_true", default=False,
                        help="Expand training set 8× with all D4 symmetry transforms (recommended with --latent_dir)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--precision", type=str, default="16-mixed", help="Training precision: 32, 16-mixed, bf16-mixed")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs). 0 = disabled.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--find_lr", action="store_true", default=False, help="Run LR range test and save plot, then exit")
    parser.add_argument("--log_dir", type=str, default="lightning_logs", help="TensorBoard log directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/unet", help="Directory to save model checkpoints")
    parser.add_argument("--latent_dir", type=str, default=None,
                        help="Path to pre-computed VAE latent .pt files. If set, skips on-the-fly VAE encoding.")
    args = parser.parse_args()

    REFERENCE_BATCH_SIZE = 2  # batch used when args.lr default was tuned
    if args.batch_size != REFERENCE_BATCH_SIZE:
        args.lr = args.lr * math.sqrt(args.batch_size / REFERENCE_BATCH_SIZE)
        print(f"[LR scaling] batch_size={args.batch_size} → lr scaled to {args.lr:.2e} (sqrt rule, ref={REFERENCE_BATCH_SIZE})")

    if args.vae_ckpt is None:
        print("WARNING: --vae_ckpt not set. The checkpoint will contain randomly-initialized "
              "VAE weights, making inference unusable. Pass --vae_ckpt to fix this.")

    # ── Config ──────────────────────────────────────────────────────────────
    if args.config is None:
        config_path = pathlib.Path(__file__).resolve().parent.parent / "opensr_model" / "configs" / "config_10m.yaml"
    else:
        config_path = pathlib.Path(args.config)
    config = OmegaConf.load(config_path)
    print(f"Config: {config_path}")

    # ── Model + Data ────────────────────────────────────────────────────────
    model = LitUNetDenoiser(config, vae_ckpt=args.vae_ckpt, lr=args.lr, max_epochs=args.epochs, warmup_epochs=args.warmup_epochs, cfg_dropout=args.cfg_dropout, no_warmup=args.no_warmup, use_precomputed=args.latent_dir is not None)
    dm = FusionDataModule(args.data_dir, batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          train_frac=args.train_frac, val_frac=args.val_frac,
                          test_frac=args.test_frac, augment=args.augment,
                          augment_d4=args.augment_d4, latent_dir=args.latent_dir)

    # ── Callbacks ───────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="unet-{epoch:04d}-{val_loss:.6f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if args.patience > 0:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=args.patience, mode="min", verbose=True))

    # ── Logger ──────────────────────────────────────────────────────────────
    logger = TensorBoardLogger(save_dir=args.log_dir, name="unet_s1s2_fusion")

    # ── Trainer ─────────────────────────────────────────────────────────────
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

    # ── LR range test ───────────────────────────────────────────────────────
    if args.find_lr:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, dm, min_lr=1e-7, max_lr=1, num_training=200)
        suggested_lr = lr_finder.suggestion()
        print(f"\nSuggested LR: {suggested_lr:.2e}")
        fig = lr_finder.plot(suggest=True)
        out_path = "lr_finder.png"
        fig.savefig(out_path)
        print(f"LR finder plot saved to: {out_path}")
        return

    # ── Train ───────────────────────────────────────────────────────────────
    trainer.fit(model, dm, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
