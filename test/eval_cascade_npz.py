"""End-to-end cascade evaluation: S1+S2 → 10m→5m UNet → 5m→1m UNet → metrics.

All inputs come from 1m-untouched NPZ files (S1, S2, and 1m aerial ground truth).
No 5m-untouched data is used.

Stage 1 (10m→5m): config_10m.yaml  — S1+S2 → predicted 5m aerial (256×256)
Stage 2 (5m→1m): two variants:
  s1s2 (default): config_5m_to_1m_with_s2.yaml  in_channels=14  VAE(5m)+S1+VAE(S2)
  s1:             config_1m.yaml                 in_channels=10  VAE(5m)+S1

Usage:
    python test/eval_cascade_npz.py \\
        --unet_ckpt_10to5 checkpoints/5m/unet/unet-epoch=0098-val_loss=0.102384.ckpt \\
        --unet_ckpt_5to1  checkpoints/5to1m_with_s2/unet/unet5to1-s1s2-epoch=0200-val_loss=0.181066.ckpt

    python test/eval_cascade_npz.py \\
        --unet_ckpt_10to5 checkpoints/5m/unet/unet-epoch=0098-val_loss=0.102384.ckpt \\
        --unet_ckpt_5to1  checkpoints/5to1m-s1/unet/unet5to1-epoch=0241-val_loss=0.176718.ckpt \\
        --variant s1 --steps_10to5 50 --out_csv results.csv
"""

import argparse
import csv
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tqdm import tqdm

import opensr_test

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opensr_model.srmodel import SRLatentDiffusion
from opensr_model.diffusion.utils import DDIMSampler
from opensr_model.utils import normalize_s1, normalize_s2, normalize_aerial

# Stage 2 constants (both s1 and s1s2 variants share the same VAE)
HR_PAD      = 1024  # 1m aerial padded to 1024×1024
HR_NATIVE   = 1000  # actual 1m content size
LATENT_SIZE = 128   # HR_PAD / 8 (vae_downscale for ch_mult [1,2,4,8])

METRIC_KEYS = ["reflectance", "spectral", "spatial", "synthesis",
               "ha_metric", "om_metric", "im_metric", "psnr", "ssim"]
METRIC_LABELS = {
    "reflectance": "Reflectance ↓",
    "spectral":    "Spectral ↓",
    "spatial":     "Spatial ↓",
    "synthesis":   "Synthesis ↑",
    "ha_metric":   "Hallucination ↓",
    "om_metric":   "Omission ↓",
    "im_metric":   "Improvement ↑",
    "psnr":        "PSNR ↑",
    "ssim":        "SSIM ↑",
}


def zero_pad(t: torch.Tensor, target: int) -> torch.Tensor:
    """Symmetric zero-pad a (C, H, W) tensor to target×target."""
    _, h, w = t.shape
    ph, pw = target - h, target - w
    return F.pad(t, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))


def load_inputs(path: pathlib.Path):
    """Load S1, S2, and 1m aerial ground truth from a 1m-untouched NPZ.

    Returns:
        s1        : (2, 128, 128) float32  dB
        s2        : (4, 128, 128) float32  DN  (padded, for model input)
        s2_raw    : (4, 100, 100) float32  DN  (native, for opensr_test LR — scale 1000/100=10)
        aerial_gt : (4, 1000, 1000) float32  [0, 255]
    """
    with np.load(path, allow_pickle=True) as d:
        s1 = torch.from_numpy(
            np.stack([d["s1_vv"].astype(np.float32), d["s1_vh"].astype(np.float32)])
        )
        s2 = torch.from_numpy(
            np.stack([d[k].astype(np.float32) for k in ("s2_r", "s2_g", "s2_b", "s2_nir")])
        )
        aerial_gt = torch.from_numpy(
            np.stack([d[k].astype(np.float32) for k in ("aerial_r", "aerial_g", "aerial_b", "aerial_nir")])
        )
    s2_raw = s2.clone()
    s1 = zero_pad(s1, 128)
    s2 = zero_pad(s2, 128)
    return s1, s2, s2_raw, aerial_gt


def load_trained_weights(model: SRLatentDiffusion, unet_ckpt: str):
    ckpt = torch.load(unet_ckpt, map_location="cpu", weights_only=False)
    remapped = {k[len("ldm."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("ldm.")}
    model.model.load_state_dict(remapped, strict=False)
    print(f"  Loaded {pathlib.Path(unet_ckpt).name}")


def build_conditioning_s1s2(model: SRLatentDiffusion,
                             aerial_5m: torch.Tensor,
                             s1: torch.Tensor,
                             s2: torch.Tensor) -> torch.Tensor:
    """10-channel conditioning for s1s2 variant: VAE(5m aerial)[4] + S1[2] + VAE(S2)[4]."""
    vae = model.model.first_stage_model
    vae_dtype = next(vae.parameters()).dtype

    aerial_5m_norm = normalize_aerial(aerial_5m, stage="norm")
    aerial_5m_up   = F.interpolate(aerial_5m_norm, size=(HR_PAD, HR_PAD), mode="bilinear", align_corners=False)

    s2_norm = normalize_s2(s2, stage="norm")
    s2_up   = F.interpolate(s2_norm, size=(HR_PAD, HR_PAD), mode="bilinear", align_corners=False)

    with torch.no_grad():
        cond_5m = vae.encode(aerial_5m_up.to(vae_dtype)).mode().float()
        cond_s2 = vae.encode(s2_up.to(vae_dtype)).mode().float()

    s1_norm = normalize_s1(s1, stage="norm")
    cond_s1 = F.interpolate(s1_norm, size=(LATENT_SIZE, LATENT_SIZE), mode="bilinear", align_corners=False)

    return torch.cat([cond_5m, cond_s1, cond_s2], dim=1)  # (B, 10, 128, 128)


def build_conditioning_s1(model: SRLatentDiffusion,
                           aerial_5m: torch.Tensor,
                           s1: torch.Tensor) -> torch.Tensor:
    """6-channel conditioning for s1 variant: VAE(5m aerial)[4] + S1[2]."""
    vae = model.model.first_stage_model
    vae_dtype = next(vae.parameters()).dtype

    aerial_5m_norm = normalize_aerial(aerial_5m, stage="norm")
    aerial_5m_up   = F.interpolate(aerial_5m_norm, size=(HR_PAD, HR_PAD), mode="bilinear", align_corners=False)

    with torch.no_grad():
        cond_5m = vae.encode(aerial_5m_up.to(vae_dtype)).mode().float()

    s1_norm = normalize_s1(s1, stage="norm")
    cond_s1 = F.interpolate(s1_norm, size=(LATENT_SIZE, LATENT_SIZE), mode="bilinear", align_corners=False)

    return torch.cat([cond_5m, cond_s1], dim=1)  # (B, 6, 128, 128)


@torch.no_grad()
def run_inference(model: SRLatentDiffusion,
                  conditioning: torch.Tensor,
                  steps: int,
                  guidance: float,
                  eta: float = 0.95,
                  temperature: float = 1.0) -> torch.Tensor:
    """DDIM sampling given pre-built conditioning. Returns (B, 4, H, W) in [0, 255]."""
    ddim = DDIMSampler(model.model)
    ddim.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=False)

    B = conditioning.shape[0]
    latent_h, latent_w = conditioning.shape[2], conditioning.shape[3]
    latent = torch.randn((B, model.z_channels, latent_h, latent_w), device=model.device)
    null_cond = torch.zeros_like(conditioning)

    time_range = np.flip(ddim.ddim_timesteps)
    for i, step in enumerate(time_range):
        index = steps - i - 1
        t = torch.full((B,), step, device=model.device, dtype=torch.long)

        if guidance > 1.0:
            e_uncond = model.model.apply_model(latent, t, cond=null_cond)
            e_cond   = model.model.apply_model(latent, t, cond=conditioning)
            e_t = e_uncond + guidance * (e_cond - e_uncond)
            latent = model._ddim_step(latent, e_t, index, ddim, temperature)
        else:
            outs = ddim.p_sample_ddim(
                x=latent, c=conditioning, t=step, index=index,
                use_original_steps=False, temperature=temperature,
            )
            latent, _ = outs

    decoded = model.model.decode_first_stage(latent)
    return normalize_aerial(decoded, stage="denorm")


def main():
    parser = argparse.ArgumentParser(
        description="Cascade eval: S1+S2 → 10m→5m UNet → 5m→1m UNet → metrics"
    )
    parser.add_argument("--unet_ckpt_10to5", type=str, required=True,
                        help="Path to 10m→5m checkpoint (config_10m.yaml)")
    parser.add_argument("--unet_ckpt_5to1", type=str, required=True,
                        help="Path to 5m→1m checkpoint")
    parser.add_argument("--variant", choices=["s1s2", "s1"], default="s1s2",
                        help="5m→1m model variant: s1s2 (default) or s1")
    parser.add_argument("--dir_1m", type=str,
                        default=str(pathlib.Path.home() / "npz/apr2025/1m-untouched"),
                        help="Directory with 1m-untouched NPZ tiles (S1+S2+1m aerial)")
    parser.add_argument("--steps_10to5", type=int,   default=100)
    parser.add_argument("--steps_5to1",  type=int,   default=100)
    parser.add_argument("--guidance_10to5", type=float, default=6.0)
    parser.add_argument("--guidance_5to1",  type=float, default=6.0)
    parser.add_argument("--eta",    type=float, default=0.95)
    parser.add_argument("--device", type=str,  default=None)
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Stage 1 — 10m→5m
    cfg1 = OmegaConf.load(ROOT / "opensr_model" / "configs" / "config_10m.yaml")
    model1 = SRLatentDiffusion(cfg1, device=device)
    print(f"Loading stage 1 checkpoint: {args.unet_ckpt_10to5}")
    load_trained_weights(model1, args.unet_ckpt_10to5)
    model1.eval()

    # Stage 2 — 5m→1m
    cfg2_name = "config_5m_to_1m_with_s2.yaml" if args.variant == "s1s2" else "config_1m.yaml"
    cfg2 = OmegaConf.load(ROOT / "opensr_model" / "configs" / cfg2_name)
    model2 = SRLatentDiffusion(cfg2, device=device)
    print(f"Loading stage 2 checkpoint ({args.variant}): {args.unet_ckpt_5to1}")
    load_trained_weights(model2, args.unet_ckpt_5to1)
    model2.eval()

    tiles = sorted(pathlib.Path(args.dir_1m).glob("*.npz"))
    print(f"Found {len(tiles)} tiles in {args.dir_1m}")

    offset = (HR_PAD - HR_NATIVE) // 2  # 12px — remove symmetric zero-padding

    rows = []
    for path in tqdm(tiles, desc="Evaluating tiles"):
        s1, s2, s2_raw, aerial_gt = load_inputs(path)

        s1 = s1.unsqueeze(0).to(device)
        s2 = s2.unsqueeze(0).to(device)

        # Stage 1: S1+S2 → predicted 5m aerial (256×256)
        aerial_pred = model1.forward(
            s2, s1,
            sampling_steps=args.steps_10to5,
            guidance_scale=args.guidance_10to5,
            sampling_eta=args.eta,
            histogram_matching=False,
            apply_nodata_mask=False,
        )  # (1, 4, 256, 256) in [0, 255]

        # Stage 2: predicted 5m aerial + S1 + S2 → 1m SR (1024×1024)
        if args.variant == "s1s2":
            cond = build_conditioning_s1s2(model2, aerial_pred, s1, s2)
        else:
            cond = build_conditioning_s1(model2, aerial_pred, s1)

        sr = run_inference(model2, cond,
                           steps=args.steps_5to1, guidance=args.guidance_5to1, eta=args.eta)

        # Crop symmetric padding → 1000×1000, compute metrics
        sr_crop = sr[0, :, offset:offset + HR_NATIVE, offset:offset + HR_NATIVE].cpu()

        sr_norm = (sr_crop / 255.0).clamp(0, 1)                 # SR aerial → [0,1], 1000×1000
        hr_norm = (aerial_gt / 255.0).clamp(0, 1)               # GT aerial → [0,1], 1000×1000
        # opensr_test requires integer scale factor; upsample LR 100→250 so scale=1000/250=4
        lr_norm = (s2_raw / 10000.0).clamp(0, 1)
        lr_os = F.interpolate(lr_norm.unsqueeze(0), size=(250, 250), mode="bilinear", align_corners=False)[0]

        result = opensr_test.Metrics().compute(lr=lr_os, sr=sr_norm, hr=hr_norm)
        row = {"tile": path.stem}
        for k in METRIC_KEYS[:7]:
            row[k] = float(result.get(k, float("nan")))

        sr_u8 = sr_norm[:3].numpy().transpose(1, 2, 0)
        gt_u8 = hr_norm[:3].numpy().transpose(1, 2, 0)
        sr_u8 = (sr_u8 * 255).clip(0, 255).astype(np.uint8)
        gt_u8 = (gt_u8 * 255).clip(0, 255).astype(np.uint8)
        row["psnr"] = float(peak_signal_noise_ratio(gt_u8, sr_u8, data_range=255))
        row["ssim"] = float(structural_similarity(gt_u8, sr_u8, channel_axis=2, data_range=255))

        rows.append(row)

    print("\n" + "=" * 52)
    print(f"  Cascade results  ({len(rows)} tiles, variant={args.variant})")
    print("=" * 52)
    for k in METRIC_KEYS:
        vals = np.array([r[k] for r in rows], dtype=float)
        mean, std = np.nanmean(vals), np.nanstd(vals)
        print(f"  {METRIC_LABELS[k]:<22}  {mean:8.4f} ± {std:.4f}")
    print("=" * 52)

    if args.out_csv and rows:
        out = pathlib.Path(args.out_csv)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["tile"] + METRIC_KEYS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-tile results saved to {out}")


if __name__ == "__main__":
    main()
