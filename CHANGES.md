# Changelog: S1+S2 Fusion Super-Resolution Project

Chronological record of all changes made to adapt the original
[opensr-model](https://github.com/ESAOpenSR/opensr-model) (LDSR-S2) for
training on Danish 5 m aerial data with Sentinel-1 + Sentinel-2 fusion
conditioning.

**Upstream baseline**: commit `9fc972a` (ESAOpenSR/opensr-model, ~170M param
latent diffusion model for Sentinel-2 4× SR to 2.5 m).

---

## Phase 1 — Data Pipeline & Normalization (March 2026)

### `opensr_model/data.py` — NEW FILE

Rewrote `FusionDataset` from scratch to handle the new NPZ tile format
produced by the Bachelor data pipeline:

- Loads **per-band keys** instead of stacked arrays:
  - S1: `s1_vv`, `s1_vh` (float32, 100×100)
  - S2: `s2_b`, `s2_g`, `s2_r`, `s2_nir` (uint16, 100×100)
  - Aerial: `aerial_r`, `aerial_g`, `aerial_b`, `aerial_nir` (uint8, 200×200)
- Stacks bands into channel-first tensors (S1: 2ch, S2: 4ch, Aerial: 4ch).
- **Zero-pads** spatial dimensions to UNet-friendly sizes:
  S1/S2 100→128 (`LR_PAD_SIZE`), Aerial 200→256 (`HR_PAD_SIZE`).
- Removed `valid` mask filtering (not present in our NPZ files).
- Added `require_aerial` and `pad` parameters.
- Added `make_train_val_datasets()` convenience function.

Initially used reflect-padding; **changed to zero-padding** to avoid training
on mirrored/fake data at tile edges.

### `opensr_model/utils.py` — MODIFIED (added 124 lines)

Added three reversible normalization functions for the fusion pipeline:

- **`normalize_s2`**: S2 RGBNIR DN → [-1, 1].
  Per-channel divisors: RGB=3000, NIR=5000.
- **`normalize_s1`**: S1 VV/VH dB → [-1, 1].
  Clipped ranges: VV [-30, 0], VH [-35, -5].
- **`normalize_aerial`**: Aerial uint8 [0, 255] → [-1, 1].

All support `stage="denorm"` for reversibility.

### `opensr_model/configs/config_10m.yaml` — MODIFIED

- Added `scale_factor: 2` (5 m aerial / 10 m S1,S2 = 2× upscaling;
  original was implicit 4× for S2→2.5 m).
- Changed UNet `in_channels: 8` → `10` (4ch noisy latent + 4ch S2 via VAE +
  2ch S1 direct = 10ch).

---

## Phase 2 — Inference Model Adaptation (March 2026)

### `opensr_model/srmodel.py` — MODIFIED (258 lines changed)

Adapted `SRLatentDiffusion` for S1+S2 dual-input fusion:

- **`_tensor_encode()`**: Rewrote from single-input S2 to dual-input S1+S2:
  - S2 (4ch) → `normalize_s2` → upsample to HR → VAE encode → 4ch latent.
  - S1 (2ch) → `normalize_s1` → upsample to latent size → 2ch direct.
  - Fuses via `torch.cat` → 6ch conditioning tensor.
  - Original only handled 4ch S2 with hardcoded 4× upscale.
- **`_tensor_decode()`**: Rewrote for aerial output:
  - Uses `normalize_aerial(denorm)` instead of `linear_transform`.
  - Histogram matching against stored S2 reference (was `self._X`, now
    `self._X_s2` with channel-count safety).
- **`_prepare_model()`**: Generalized latent shape to use `self.z_channels`
  and conditioning spatial dims (was hardcoded).
- **`forward()`**: Updated signature to accept `X_s2` and `X_s1` separately
  (was single `X` input).
- **Added dynamic config attributes**: `self.scale_factor`,
  `self.vae_downscale`, `self.z_channels` — computed from config instead of
  hardcoded.

### `test/test_1.py` — NEW FILE

End-to-end inference smoke test: loads an NPZ file, runs
`SRLatentDiffusion.forward()` with random weights, saves SR output + aerial
GT + S2 input as PNGs.

---

## Phase 3 — Training Scripts: PyTorch Lightning (March 2026)

### `train/train_unet.py` — NEW FILE (replaces root-level `train_unet.py`)

Complete rewrite of the UNet training script to PyTorch Lightning:

- **`LitUNetDenoiser(pl.LightningModule)`**:
  - Builds `LatentDiffusion` model (UNet + VAE).
  - `_load_vae()`: Loads pretrained VAE checkpoint, strips `vae.` and
    `disc.`/`lpips_fn.` key prefixes, uses `strict=True`.
  - Freezes all VAE parameters; only UNet trains.
  - `_build_conditioning()`: S2→VAE encode + S1→upsample → 6ch conditioning.
  - `_encode_aerial()`: Aerial→normalize→VAE encode → target latent z_0.
  - `_shared_step()`: Full diffusion step (z_0, conditioning, noise, UNet
    predict, MSE loss).
  - `configure_optimizers()`: AdamW + LinearLR warmup + CosineAnnealingLR.
  - `on_train_epoch_start()`: Ensures VAE stays in eval mode.
- **`AugmentedDataset`**: Random horizontal/vertical flip + 90° rotations
  applied identically to S1, S2, and aerial.
- **`FusionDataModule(pl.LightningDataModule)`**:
  - 80/10/10 train/val/test split via `random_split` (seed=42).
  - Optional augmentation via `--augment` flag.
- **CLI features**: `--vae_ckpt`, `--precision 16-mixed`, `--devices N`,
  `--augment`, `--cfg_dropout 0.15` (classifier-free guidance),
  `--warmup_epochs`, `--patience` (EarlyStopping), `--find_lr` (LR range
  test), `--resume` (checkpoint resume).

**Removed from original**: Manual training loop, MLflow integration, manual
device management, `min_valid_frac`.

### `train/train_vae.py` — NEW FILE

VAE training script implementing the full LDSR-S2 paper loss (Eq. 1):

```
L_total = λ_WD · MMD(z, z_prior) + λ_MAE · L1(x̂, x) + λ_GAN · hinge(x̂) + λ_LPIPS · LPIPS(x̂, x)
```

- **`LitVAE(pl.LightningModule)`** with manual optimization (dual-optimizer
  GAN setup):
  - Wasserstein Distance via MMD with RBF kernel (multi-bandwidth).
  - `PatchDiscriminator` (4ch RGBNIR, hinge loss, activated after warmup).
  - LPIPS (VGG backbone, frozen) with random 3-of-4 band selection per step.
  - Gradient clipping at 1.0 for both VAE and discriminator.
- **`AerialDataModule`**: 95/5/0 train/val/test split (VAE sees nearly all
  data; split only matters for UNet evaluation).
- **MMD implementation**: RBF kernel at 7 bandwidths, fp32 cast + clamping
  for numerical stability under fp16 training, batch-size guard (B<2→0).

### `train/train_unet_old.py` — PRESERVED

Backup of the pre-Lightning manual training loop for reference.

---

## Phase 4 — Server Deployment & Bug Fixes (March–April 2026)

### Server setup (aragorn.netlab.eng.au.dk)

- Created `.venv` on server with CUDA-enabled PyTorch.
- Installed `tensorboard` (missing from requirements).
- Resolved CUDA OOM on 10.75 GB GPU by reducing batch size.

### `opensr_model/autoencoder/autoencoder.py` — MODIFIED (server)

Added **gradient checkpointing** in Encoder and Decoder forward passes to
reduce VRAM usage (~30–40% savings):

- `self.down[i_level].block[i_block](h, temb)` →
  `checkpoint(self.down[i_level].block[i_block], h, temb, use_reentrant=False)`
- Applied to all ResnetBlocks in encoder down-blocks, mid-blocks, and decoder
  up-blocks/mid-blocks.
- **No architectural changes** — weights are fully compatible between
  checkpointed and non-checkpointed versions.

### VAE training completed (server)

Trained AutoencoderKL on ~all NPZ tiles (95/5 split) with full paper loss.
Final MAE ≈ 0.04. Checkpoint saved to `checkpoints/vae/last.ckpt`.

### Critical bug fix: `_load_vae()` key prefix mismatch

**Bug**: VAE checkpoint keys had `vae.` prefix (e.g. `vae.encoder.conv_in.weight`)
but `_load_vae()` only stripped `module.` prefix. With `strict=False`, this
caused **zero keys to be loaded** — VAE ran with random weights → UNet
produced pure noise output.

**Diagnosis**: Ran key comparison script that showed `Matched keys: 0` with
`module.` stripping, but `204 matched` after removing `vae.` prefix.

**Fix**: Updated `_load_vae()` in `train/train_unet.py`:
- Strip `vae.` prefix from checkpoint keys.
- Filter out `disc.*` and `lpips_fn.*` keys (discriminator/LPIPS weights).
- Changed to `strict=True` so mismatches fail loudly.
- Prints match statistics for verification.

After fix: `VAE loaded: 204 keys (missing: 0, unexpected: 0)` ✅

### `opensr_model/srmodel.py` — Restored `_tensor_decode()`

Method was accidentally deleted during whitespace cleanup. Restored with
updated logic (uses `normalize_aerial` denorm and `self._X_s2` reference).

---

## Phase 5 — Documentation (March–April 2026)

### `CHANGES.md` — NEW FILE (this file)

Chronological record of all project changes.

### `flow.md` — NEW FILE

Detailed step-by-step documentation of how a single NPZ tile flows through
the entire pipeline: data loading → VAE training → UNet training → inference.
Includes tensor shapes at every stage, ASCII architecture diagrams, and model
parameter breakdown.

### `README.md` — MODIFIED

Added training & server deployment sections (§5.2–5.3).

---

## Dependencies (`requirements.txt`) — MODIFIED

- Updated all packages to modern, non-pinned versions (`>=`).
- Added: `pytorch-lightning>=2.5.0`, `lpips>=0.1.4`.
- Removed: duplicate `torch` entry, `taming-transformers`, `wandb`,
  `pathtools`, `oauthlib`, `torchaudio`.
- Replaced `opensr_utils` with `requests`.

---

## Files Summary

### New files (our contributions)
| File | Purpose |
|---|---|
| `opensr_model/data.py` | FusionDataset for S1+S2+Aerial NPZ tiles |
| `train/train_vae.py` | VAE training with full paper loss (WD+MAE+GAN+LPIPS) |
| `train/train_unet.py` | UNet denoiser training (Lightning) |
| `train/train_unet_old.py` | Pre-Lightning training script backup |
| `test/test_1.py` | Inference smoke test |
| `CHANGES.md` | This changelog |
| `flow.md` | Pipeline flow documentation |

### Modified files
| File | Change |
|---|---|
| `opensr_model/srmodel.py` | S1+S2 fusion encode/decode, dynamic config |
| `opensr_model/utils.py` | Added normalize_s1/s2/aerial functions |
| `opensr_model/configs/config_10m.yaml` | scale_factor=2, in_channels=10 |
| `opensr_model/autoencoder/autoencoder.py` | Gradient checkpointing (server) |
| `requirements.txt` | Modernized dependencies |
| `README.md` | Training & deployment docs |

### Unchanged files
| File | Note |
|---|---|
| `opensr_model/diffusion/` | Diffusion framework (LatentDiffusion, DDPM, DDIM) untouched |
| `opensr_model/denoiser/` | UNet architecture untouched |
| `opensr_model/autoencoder/utils.py` | ResnetBlock, attention, etc. untouched |

---

## Pipeline Dimensions (verified)

```
Input:          s1 (B, 2, 128, 128)   s2 (B, 4, 128, 128)   aerial (B, 4, 256, 256)
Aerial latent:  z_0 (B, 4, 64, 64)
Conditioning:   (B, 6, 64, 64)  = 4ch VAE(S2) + 2ch S1
UNet input:     (B, 10, 64, 64) = 4ch z_t + 6ch conditioning
Noise pred:     (B, 4, 64, 64)
SR output:      (B, 4, 200, 200) after removing padding
```

## Model Size

```
VAE Encoder:       22,361,608 params
VAE Decoder:       32,966,532 params
VAE Total:         55,328,232 params  (32.7%)
UNet Denoiser:    113,629,764 params  (67.3%)
Inference Total:  168,957,996 params  (~170M)
PatchGAN Disc:      2,766,657 params  (VAE training only)
```
