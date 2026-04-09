# Data Flow: From NPZ Tile to Super-Resolution Output

This document describes the complete journey of a single NPZ tile through the
Latent Diffusion Super-Resolution pipeline — from raw sensor data on disk to a
5 m aerial super-resolution image.

The pipeline has three phases that are executed sequentially:

1. **Phase 1 — VAE Training** (`train/train_vae.py`): Train the autoencoder on aerial images only.
2. **Phase 2 — UNet Training** (`train/train_unet.py`): Freeze VAE, train the diffusion denoiser.
3. **Inference** (`opensr_model/srmodel.py`): Given only S1+S2, produce a super-resolved aerial image.

---

## Step 0: Raw Data on Disk

Each `.npz` file represents a **1 km × 1 km tile** and contains 10 separate 2D
arrays from three co-registered sensors:

```
Sentinel-1 (Radar, 10 m resolution):
  s1_vv       : (100, 100) float32  — VV backscatter in dB
  s1_vh       : (100, 100) float32  — VH backscatter in dB

Sentinel-2 (Optical, 10 m resolution):
  s2_b        : (100, 100) uint16   — Band 02 (Blue), DN values
  s2_g        : (100, 100) uint16   — Band 03 (Green), DN values
  s2_r        : (100, 100) uint16   — Band 04 (Red), DN values
  s2_nir      : (100, 100) uint16   — Band 08 (NIR), DN values

Aerial Orthophoto (5 m resolution):
  aerial_r    : (200, 200) uint8    — Red
  aerial_g    : (200, 200) uint8    — Green
  aerial_b    : (200, 200) uint8    — Blue
  aerial_nir  : (200, 200) uint8    — Near-Infrared
```

S1/S2 cover the same area at 10 m → 100 pixels.
Aerial covers the same area at 5 m → 200 pixels (2× the spatial resolution).

**Source file**: `opensr_model/data.py`, lines 1–28.

---

## Step 1: Data Loading and Padding

**File**: `opensr_model/data.py` → `FusionDataset.__getitem__()` (line 130)

### 1a. Load and stack per-band arrays into channel-first tensors

```python
s1 = np.stack([npz["s1_vv"], npz["s1_vh"]])          # → (2, 100, 100)
s2 = np.stack([npz["s2_b"], npz["s2_g"],
               npz["s2_r"], npz["s2_nir"]])           # → (4, 100, 100)
aerial = np.stack([npz["aerial_r"], npz["aerial_g"],
                   npz["aerial_b"], npz["aerial_nir"]])  # → (4, 200, 200)
```

All arrays are cast to `float32`. Note that S2 values are raw DN (digital
numbers, typically 0–3000 for RGB, 0–5000 for NIR) and aerial values are raw
pixel intensities (0–255).

### 1b. Zero-pad to UNet-friendly sizes

The UNet architecture requires spatial dimensions divisible by `2^n` where `n`
is the number of downsampling steps. We zero-pad to the next valid size:

```
s1     : (2, 100, 100) → (2, 128, 128)   +14 px border of zeros
s2     : (4, 100, 100) → (4, 128, 128)   +14 px border of zeros
aerial : (4, 200, 200) → (4, 256, 256)   +28 px border of zeros
```

**Why zero-pad (not reflect-pad)?** Reflect-padding creates mirrored data at
the edges that the model could learn as real features. Zero-padding is
transparent — the model can learn to ignore it.

**Source file**: `opensr_model/data.py`, `_zero_pad()` at line 112.

### Output from Step 1

```
batch["s1"]     : (B, 2, 128, 128)  float32   — raw dB values
batch["s2"]     : (B, 4, 128, 128)  float32   — raw DN values
batch["aerial"] : (B, 4, 256, 256)  float32   — raw pixel values [0, 255]
```

---

## Phase 1: VAE Training (only aerial images)

**File**: `train/train_vae.py` → `LitVAE`

The VAE learns to compress and reconstruct aerial RGBNIR images. S1 and S2 are
**not used** in this phase.

### Step 2A: Normalize aerial to [-1, 1]

**File**: `opensr_model/utils.py` → `normalize_aerial()` (line 111)

```
aerial: (B, 4, 256, 256)  values [0, 255]
  ↓  divide by 255        → [0, 1]
  ↓  multiply by 2, sub 1 → [-1, 1]
x: (B, 4, 256, 256)       values [-1, +1]
```

All neural network operations happen in the [-1, 1] range. This is standard
for generative models — it centers the data around zero, making optimization
easier and activation functions more effective.

### Step 3A: Encoder — compress image to latent space

**File**: `opensr_model/autoencoder/autoencoder.py` → `Encoder.forward()` (line 126), called via `AutoencoderKL.encode()` (line 536)

The encoder is a series of ResNet blocks with downsampling:

```
x: (B, 4, 256, 256)        — normalized aerial image
  ↓ conv_in: 4ch → 128ch   — initial spectral expansion
h: (B, 128, 256, 256)

  ↓ 2× ResBlock             — ch_mult[0] = 1 → 128ch
  ↓ Downsample (÷2)
h: (B, 128, 128, 128)

  ↓ 2× ResBlock             — ch_mult[1] = 2 → 256ch
  ↓ Downsample (÷2)
h: (B, 256, 64, 64)

  ↓ 2× ResBlock             — ch_mult[2] = 4 → 512ch
  (no downsample — last level)
h: (B, 512, 64, 64)

  ↓ mid_block_1 (ResBlock)
  ↓ mid_attn (self-attention)  — global context at 64×64
  ↓ mid_block_2 (ResBlock)
h: (B, 512, 64, 64)

  ↓ GroupNorm + Swish + conv_out
h: (B, 8, 64, 64)            — 8 channels because double_z=True (2×4)
```

Then `quant_conv` (a 1×1 convolution) maps to the final 8-channel output:

```
moments: (B, 8, 64, 64)
```

### Step 4A: Gaussian distribution — split into mean and variance

**File**: `opensr_model/autoencoder/autoencoder.py` → `DiagonalGaussianDistribution` (line 353)

The 8 channels are split in half:

```
moments: (B, 8, 64, 64)
  ↓ chunk(2, dim=1)
mean:   (B, 4, 64, 64)   — the "best guess" latent representation
logvar: (B, 4, 64, 64)   — log-variance (how uncertain the encoding is)
```

The logvar is clamped to [-30, 20] for numerical stability.

### Step 5A: Sample from the posterior

```python
z = mean + std × ε      where ε ~ N(0, I)
```

- `std = exp(0.5 × logvar)`
- `z: (B, 4, 64, 64)`

**This is the latent representation.** The image has been compressed from
256×256×4 = 262,144 values down to 64×64×4 = 16,384 values. A **16× compression**.

### Step 6A: Wasserstein Distance (MMD regularization)

**File**: `train/train_vae.py` → `mmd_loss()` (line 99)

The VAE's latent space must be well-structured for the diffusion model to work
in it later. We enforce this by penalizing the difference between the encoded
distribution Q(Z) and a standard normal prior P(Z) = N(0, I).

Instead of KL divergence (as in standard VAEs), we use the **Wasserstein
Distance** approximated via **Maximum Mean Discrepancy (MMD)** with RBF kernels:

```python
z_prior = torch.randn_like(z)       # samples from N(0, I)
wd_loss = mmd_loss(z, z_prior)      # MMD between encoded and prior
```

**Why Wasserstein instead of KL?**
- KL forces *each individual* image's latent to look like N(0,1) → latent
  codes overlap → blurry reconstructions.
- Wasserstein forces only the *aggregate* distribution to match N(0,1) →
  individual codes can stay separated → sharper reconstructions.

The MMD is computed via an RBF kernel at multiple bandwidths (0.1, 0.2, 0.5,
1.0, 2.0, 5.0, 10.0), which captures both local and global distributional
differences. The formula (from the WAE paper, Algorithm 2):

```
MMD² = Σ k(z_i, z_j) / n(n-1)         [encoded vs encoded]
     + Σ k(z̃_i, z̃_j) / n(n-1)        [prior vs prior]
     - 2 Σ k(z_i, z̃_j) / n²           [cross-term]
```

If MMD = 0, the two distributions are identical.

**Reference**: Tolstikhin et al., "Wasserstein Auto-Encoders" (ICLR 2018).

### Step 7A: Decoder — reconstruct image from latent

**File**: `opensr_model/autoencoder/autoencoder.py` → `Decoder.forward()` (line 306), called via `AutoencoderKL.decode()` (line 559)

The decoder is the mirror of the encoder:

```
z: (B, 4, 64, 64)          — sampled latent
  ↓ post_quant_conv: 4→512ch
h: (B, 512, 64, 64)

  ↓ mid_block_1 + attention + mid_block_2
h: (B, 512, 64, 64)

  ↓ 3× ResBlock + Upsample (×2)   → 512→256ch
h: (B, 256, 128, 128)

  ↓ 3× ResBlock + Upsample (×2)   → 256→128ch
h: (B, 128, 256, 256)

  ↓ 3× ResBlock (no upsample)     → 128ch
  ↓ GroupNorm + Swish + conv_out → 4ch
reconstruction: (B, 4, 256, 256)   — values in [-1, +1]
```

### Step 8A: Loss computation (paper Eq. 1)

**File**: `train/train_vae.py` → `LitVAE._vae_loss()` (line 249)

```
L_total = λ_WD    · MMD(z, z_prior)                      [latent regularization]
        + λ_MAE   · L1(reconstruction, x)                 [pixel reconstruction]
        + λ_GAN   · hinge_loss_g(disc(reconstruction))    [realism, after warmup]
        + λ_LPIPS · LPIPS(reconstruction[:,3ch], x[:,3ch]) [perceptual quality]
```

| Loss component | Purpose | Default λ |
|---|---|---|
| **WD (MMD)** | Latent space → N(0,I) structure | 1.0 |
| **MAE** | Pixel-accurate reconstruction | 1.0 |
| **GAN** | Sharp, realistic textures | 0.5 |
| **LPIPS** | Perceptual similarity (VGG features) | 1.0 |

The GAN discriminator (`PatchDiscriminator`) is trained in parallel after a
warm-up period (default 10 epochs). LPIPS randomly selects 3 of 4 bands each
step because VGG expects 3-channel input (paper Section IV-A).

**Backpropagation updates encoder + decoder weights.**

### VAE training is complete. Freeze all weights.

---

## Phase 2: UNet Training (VAE frozen)

**File**: `train/train_unet.py` → `LitUNetDenoiser`

Now we use **all three sensors**: S1, S2, and aerial. The VAE is frozen — its
weights never change again.

### Step 2B: Encode aerial → target latent z_0

**File**: `train/train_unet.py` → `_encode_aerial()` (line 127)

```
aerial: (B, 4, 256, 256)  [0–255]
  ↓ normalize_aerial()    → [-1, 1]
  ↓ VAE encoder (frozen)  → posterior
  ↓ sample                → z_0: (B, 4, 64, 64)
```

**z_0 is the "answer"** — the latent code that the UNet must learn to produce
from S1+S2 input alone. During inference, we won't have the aerial image, so
the UNet must generate this from scratch.

### Step 3B: Build conditioning from S1 + S2

**File**: `train/train_unet.py` → `_build_conditioning()` (line 109)

**S2 pathway** (optical, 4 channels → 4ch latent):

```
s2: (B, 4, 128, 128)           — raw DN values
  ↓ normalize_s2()             → [-1, 1]
  ↓ F.interpolate bilinear     → (B, 4, 256, 256)   upsample to HR size
  ↓ VAE encoder (frozen)       → posterior → sample
cond_s2: (B, 4, 64, 64)        — S2 in latent space
```

S2 goes through the **same VAE** as the aerial image. This is a key insight
from the LDSR-S2 paper: by encoding S2 into the same latent space as the target
aerial, the UNet doesn't need to learn the pixel→latent mapping during
denoising. This leads to better spectral consistency.

**S1 pathway** (radar, 2 channels → 2ch direct):

```
s1: (B, 2, 128, 128)           — raw dB values
  ↓ normalize_s1()             → [-1, 1]
  ↓ F.interpolate bilinear     → (B, 2, 64, 64)    upsample to latent size
cond_s1: (B, 2, 64, 64)        — S1 at latent resolution
```

S1 does **not** go through the VAE. It's a completely different modality (radar
vs optical) and the 4-channel VAE cannot process 2-channel radar data. Instead,
it's directly upsampled to the latent spatial size.

**Fusion**:

```python
conditioning = torch.cat([cond_s2, cond_s1], dim=1)
# → (B, 6, 64, 64)  =  4ch S2 latent + 2ch S1
```

### Step 4B: Forward diffusion — add noise to target

**File**: `train/train_unet.py` → `_shared_step()` (line 137)

```python
t = torch.randint(0, 1000, (B,))       # random timestep, e.g. t=347
noise = torch.randn_like(z_0)          # ε ~ N(0, I)

z_t = sqrt(ᾱ_t) · z_0 + sqrt(1 - ᾱ_t) · noise
```

Where `ᾱ_t` is the cumulative product of the noise schedule (controlled by
`linear_start=0.0001` and `linear_end=0.0155` in `config_10m.yaml`).

- At t=0: `ᾱ_t ≈ 1.0` → z_t ≈ z_0 (almost no noise)
- At t=999: `ᾱ_t ≈ 0.0` → z_t ≈ noise (almost pure noise)

`z_t` is a controlled mixture of the real aerial latent and random noise.

### Step 5B: UNet predicts the noise

```python
unet_input = cat(z_t, conditioning)   # (B, 10, 64, 64) = 4ch noisy + 6ch cond
noise_pred = UNet(unet_input, t)      # (B, 4, 64, 64)  = predicted noise
```

The UNet architecture:
- `in_channels=10` (4 noisy latent + 6 conditioning)
- `out_channels=4` (predicted noise, same shape as latent)
- `model_channels=160`, `channel_mult=[1,2,2,4]`
- Self-attention at resolutions [16, 8]
- Timestep `t` is injected via sinusoidal positional embedding

The UNet sees the noisy image and the S1+S2 conditioning and must predict:
"What is the noise component in this noisy latent?"

### Step 6B: Loss

```python
loss = F.mse_loss(noise_pred, noise)
```

Simple mean squared error between predicted noise and actual noise.
Backpropagation updates **only UNet weights** (VAE is frozen).

### UNet training is complete. Both models are now ready for inference.

---

## Inference (no aerial — only S1 + S2 as input)

**File**: `opensr_model/srmodel.py` → `SRLatentDiffusion.forward()` (line 175)

This is the production use case: given a new location with S1 and S2 data but
**no aerial imagery**, produce a super-resolved 5 m aerial-quality image.

### Step 1C: Load and pad S1 + S2

Same as Step 1, but only `s1` and `s2` are used (aerial is not available).

### Step 2C: Build conditioning

**File**: `opensr_model/srmodel.py` → `_tensor_encode()` (line 72)

Identical to Step 3B:

```
s2 → normalize → upsample to 256×256 → VAE encode → cond_s2: (B, 4, 64, 64)
s1 → normalize → upsample to 64×64                → cond_s1: (B, 2, 64, 64)
conditioning = cat(cond_s2, cond_s1) → (B, 6, 64, 64)
```

### Step 3C: Initialize with pure noise

**File**: `opensr_model/srmodel.py` → `_prepare_model()` (line 146)

```python
latent = torch.randn(B, 4, 64, 64)   # pure Gaussian noise
```

This is where the generation starts — from **nothing but randomness**.

### Step 4C: DDIM denoising loop (100 steps)

**File**: `opensr_model/srmodel.py` → `forward()` loop at line 230

```python
for i, step in enumerate(time_range):     # 100 steps, t=990→0
    noise_pred = UNet(cat(latent, conditioning), t=step)
    latent = ddim_step(latent, noise_pred, t)   # remove predicted noise
```

At each step:
1. Concatenate current latent (4ch) with conditioning (6ch) → 10ch
2. UNet predicts the noise component at this timestep
3. DDIM removes the predicted noise, producing a slightly cleaner latent
4. Repeat with the next (lower) timestep

```
Step 100: latent = pure noise             (B, 4, 64, 64)
Step  99: UNet removes a bit of noise     — guided by S1+S2
Step  98: slightly cleaner
  ...
Step   1: almost clean
Step   0: latent = denoised aerial latent (B, 4, 64, 64)
```

The conditioning is provided at **every step**, continuously guiding the
denoising toward a latent that is consistent with the S1+S2 observations.

### Step 5C: VAE decode → super-resolution image

**File**: `opensr_model/srmodel.py` → `_tensor_decode()` (line 132)

```
latent: (B, 4, 64, 64)          — denoised latent from DDIM
  ↓ VAE decoder (frozen)
decoded: (B, 4, 256, 256)        — values in [-1, +1]
  ↓ normalize_aerial(denorm)
sr: (B, 4, 256, 256)             — values in [0, 255]
  ↓ histogram matching with S2   — spectral correction
  ↓ clamp negatives → 0
  ↓ revert_padding               — remove zero-pad border
sr: (B, 4, 200, 200)             — final super-resolution RGBNIR
```

The histogram matching step aligns the spectral distribution of the output
with the input S2 image, ensuring spectral consistency (important for
downstream remote sensing applications like NDVI computation).

---

## Summary Diagram

```
NPZ File on Disk
├── s1_vv, s1_vh           (100×100, radar, dB)
├── s2_b, s2_g, s2_r, s2_nir  (100×100, optical, DN)
└── aerial_r/g/b/nir       (200×200, aerial, uint8)
         │
         ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 1: Load + Stack + Zero-Pad                        │
│    s1:     (2, 128, 128)                                │
│    s2:     (4, 128, 128)                                │
│    aerial: (4, 256, 256)                                │
└────────────────────────┬────────────────────────────────┘
                         │
    ╔════════════════════╧════════════════════════════════╗
    ║  PHASE 1: VAE Training  (train/train_vae.py)       ║
    ║  Uses: aerial only                                  ║
    ║                                                     ║
    ║  aerial [0,255] → normalize [-1,1]                  ║
    ║    ↓                                                ║
    ║  Encoder: (4, 256, 256) → (512, 64, 64) → (8, 64, 64) ║
    ║    ↓                                                ║
    ║  Split → mean (4, 64, 64) + logvar (4, 64, 64)     ║
    ║    ↓                                                ║
    ║  Sample: z = mean + std × ε    → z (4, 64, 64)     ║
    ║    ↓                                                ║
    ║  Decoder: z (4, 64, 64) → reconstruction (4, 256, 256) ║
    ║    ↓                                                ║
    ║  Loss = λ_WD·MMD(z) + λ_MAE·L1 + λ_GAN·GAN + λ_LPIPS·LPIPS ║
    ║  → Update encoder + decoder weights                 ║
    ╚═════════════════════════════════════════════════════╝
                         │
                   [Freeze VAE]
                         │
    ╔════════════════════╧════════════════════════════════╗
    ║  PHASE 2: UNet Training  (train/train_unet.py)     ║
    ║  Uses: s1, s2, aerial  (VAE frozen)                 ║
    ║                                                     ║
    ║  aerial → normalize → VAE encode → z_0 (target)    ║
    ║                                                     ║
    ║  s2 → norm → upsample 256 → VAE encode → cond_s2 (4ch) ║
    ║  s1 → norm → upsample  64 (direct)    → cond_s1 (2ch) ║
    ║  conditioning = cat(cond_s2, cond_s1)   → (6, 64, 64)  ║
    ║                                                     ║
    ║  t ~ Uniform(0, 1000)                               ║
    ║  ε ~ N(0, I)                                        ║
    ║  z_t = √ᾱ_t · z_0 + √(1-ᾱ_t) · ε                ║
    ║                                                     ║
    ║  UNet(cat(z_t, cond), t) → noise_pred (4, 64, 64)  ║
    ║  Loss = MSE(noise_pred, ε)                          ║
    ║  → Update UNet weights only                         ║
    ╚═════════════════════════════════════════════════════╝
                         │
                   [Both models ready]
                         │
    ╔════════════════════╧════════════════════════════════╗
    ║  INFERENCE  (opensr_model/srmodel.py)               ║
    ║  Input: s1, s2 only  (no aerial)                    ║
    ║                                                     ║
    ║  s2 + s1 → conditioning (6, 64, 64)                 ║
    ║  latent = randn(4, 64, 64)     ← pure noise        ║
    ║                                                     ║
    ║  for step in [100, 99, ..., 1, 0]:                  ║
    ║      noise_pred = UNet(cat(latent, cond), t=step)   ║
    ║      latent = DDIM_step(latent, noise_pred)         ║
    ║                                                     ║
    ║  latent → VAE decode → (4, 256, 256)                ║
    ║  → denormalize [0, 255]                             ║
    ║  → histogram matching                               ║
    ║  → remove padding                                   ║
    ║  → SR output: (4, 200, 200) aerial RGBNIR           ║
    ╚═════════════════════════════════════════════════════╝
```

---

## Tensor Shape Summary

| Stage | Tensor | Shape | Values |
|---|---|---|---|
| Raw NPZ | s1 bands | (100, 100) × 2 | dB, e.g. -25.3 |
| Raw NPZ | s2 bands | (100, 100) × 4 | DN, e.g. 1205 |
| Raw NPZ | aerial bands | (200, 200) × 4 | uint8, 0–255 |
| After load+pad | s1 | (B, 2, 128, 128) | float32, dB |
| After load+pad | s2 | (B, 4, 128, 128) | float32, DN |
| After load+pad | aerial | (B, 4, 256, 256) | float32, 0–255 |
| After normalize | s1 | (B, 2, 128, 128) | [-1, +1] |
| After normalize | s2 | (B, 4, 128, 128) | [-1, +1] |
| After normalize | aerial | (B, 4, 256, 256) | [-1, +1] |
| VAE encoder output | moments | (B, 8, 64, 64) | unbounded |
| Latent (z) | z | (B, 4, 64, 64) | ~ N(0, 1) |
| S2 conditioning | cond_s2 | (B, 4, 64, 64) | ~ N(0, 1) |
| S1 conditioning | cond_s1 | (B, 2, 64, 64) | [-1, +1] |
| Fused conditioning | conditioning | (B, 6, 64, 64) | mixed |
| UNet input | cat(z_t, cond) | (B, 10, 64, 64) | mixed |
| UNet output | noise_pred | (B, 4, 64, 64) | ~ N(0, 1) |
| VAE decoder output | reconstruction | (B, 4, 256, 256) | [-1, +1] |
| Final SR output | sr | (B, 4, 200, 200) | [0, 255] |

---

## Model Parameters

| Component | Parameters | Role |
|---|---|---|
| VAE Encoder | 22,361,608 | Compress 256×256 → 64×64 latent |
| VAE Decoder | 32,966,532 | Reconstruct 64×64 → 256×256 |
| VAE (total) | 55,328,232 (32.7%) | Image compression/decompression |
| UNet Denoiser | 113,629,764 (67.3%) | Noise prediction in latent space |
| **Inference total** | **168,957,996** | **~170M parameters** |
| PatchGAN Discriminator | 2,766,657 | VAE training only (discarded) |
| LPIPS (VGG) | ~14M (frozen) | VAE training only (not saved) |

Memory footprint: **0.68 GB** (FP32), **0.34 GB** (FP16).

---

## Key Design Decisions

1. **Wasserstein Distance (MMD) instead of KL divergence**: Produces a
   better-structured latent space with sharper reconstructions. Based on the
   WAE paper (Tolstikhin et al., 2018).

2. **S2 encoded through VAE, S1 not**: S2 is optical (same modality as
   aerial) and benefits from sharing the latent space. S1 is radar (completely
   different signal type) and is injected directly.

3. **Conditioning encoded before diffusion**: Following the LDSR-S2 paper,
   encoding S2 into latent space before feeding it to the UNet removes the
   burden of pixel→latent translation from the denoiser, leading to better
   spectral consistency.

4. **Zero-padding instead of reflect-padding**: Prevents the model from
   learning mirrored edge artifacts as real features.

5. **Gradient checkpointing in autoencoder**: Trades ~20–30% compute time
   for ~30–40% VRAM savings, enabling training on GPUs with limited memory.

---

## References

- **LDSR-S2 paper**: Donike et al., "Trustworthy Super-Resolution of
  Multispectral Sentinel-2 Imagery With Latent Diffusion", IEEE JSTARS 2025.
- **WAE paper**: Tolstikhin et al., "Wasserstein Auto-Encoders", ICLR 2018.
- **Original LDM**: Rombach et al., "High-Resolution Image Synthesis with
  Latent Diffusion Models", CVPR 2022.
- **DDIM**: Song et al., "Denoising Diffusion Implicit Models", ICLR 2021.
