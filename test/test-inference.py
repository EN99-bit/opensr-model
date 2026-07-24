"""Inference test using locally trained VAE + UNet checkpoints.

Loads the UNet Lightning checkpoint (which contains both frozen VAE weights
and trained UNet weights under the 'ldm.*' prefix) into SRLatentDiffusion,
then runs inference on the test NPZ tile and saves a single side-by-side
comparison panel (same layout as test-inference-batch.py):

    S1 (VV/VH) | S2 input | SR prediction | Aerial GT

Usage:
    python test/test-inference.py
    python test/test-inference.py --unet_ckpt checkpoints/unet/best.ckpt

    # S2-only model (no S1), e.g. the unet-no-s1 checkpoint:
    python test/test-inference.py \
        --unet_ckpt checkpoints/5m/unet-no-s1/last.ckpt \
        --config opensr_model/configs/config_10m_no_s1.yaml \
        --include s2
"""

import argparse
import pathlib
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opensr_model.srmodel import SRLatentDiffusion
from opensr_model.data import (FusionDataset, LR_PAD_SIZE, HR_PAD_SIZE,
                               LR_NATIVE, HR_NATIVE, LR_PAD, HR_PAD)
from opensr_model.utils import normalize_s2

ORIG_LR = LR_NATIVE   # 100
ORIG_HR = HR_NATIVE   # 1000


def tensor_to_rgb(t):
    """Convert (1,C,H,W) or (C,H,W) tensor to (H,W,3) uint8 (bands 0,1,2)."""
    if t.dim() == 4:
        t = t[0]
    img = np.clip(t[:3, :, :].cpu().numpy(), 0, 255).astype(np.uint8)
    return np.transpose(img, (1, 2, 0))


def tensor_to_s1_rgb(t):
    """Convert S1 (1,2,H,W) dB tensor to (H,W,3) uint8 via 2–98% stretch (R=VV, G=VH, B=VV)."""
    vv, vh = t[0, 0].cpu().numpy(), t[0, 1].cpu().numpy()

    def stretch(arr):
        lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
        return (np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)

    vv_n, vh_n = stretch(vv), stretch(vh)
    return np.stack([vv_n, vh_n, vv_n], axis=-1)

TEST_DIR = pathlib.Path(__file__).parent
DEFAULT_UNET_CKPT = ROOT / "checkpoints" / "1m" / "unet" / "last.ckpt"


def load_trained_weights(model: SRLatentDiffusion, unet_ckpt: str):
    """Load trained weights from a LitUNetDenoiser Lightning checkpoint.

    The checkpoint stores all weights under the 'ldm.*' prefix (self.ldm in
    LitUNetDenoiser). SRLatentDiffusion.model is the LatentDiffusion directly,
    so we strip that prefix. The checkpoint includes both the frozen VAE and
    trained UNet, so no separate VAE checkpoint is needed.
    """
    ckpt = torch.load(unet_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    remapped = {}
    for k, v in state_dict.items():
        if k.startswith("ldm."):
            remapped[k[len("ldm."):]] = v

    missing, unexpected = model.model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print(f"  Loaded {len(remapped)} keys from {unet_ckpt}")


def make_s2_only_encode(model: SRLatentDiffusion):
    """Build a true S2-only conditioning encoder (4ch, no S1 concat).

    Mirrors the S2 branch of SRLatentDiffusion._tensor_encode exactly, but stops
    before fusing S1. This matches how the no-S1 UNet (config_10m_no_s1.yaml) was
    trained: conditioning = VAE(S2) only. Unlike eval.py's `--include s2`, which
    zeros the S1 channels of a 6ch conditioning, this returns genuine 4ch
    conditioning for an 8ch UNet.
    """
    def _encode(X_s2: torch.Tensor, X_s1: torch.Tensor = None):
        model._X_s2 = X_s2.clone()
        lr_size = X_s2.shape[-1]
        hr_size = lr_size * model.scale_factor
        latent_size = hr_size // model.vae_downscale
        X_s2_norm = normalize_s2(X_s2, stage="norm")
        if model.encode_conditioning:
            X_s2_up = torch.nn.functional.interpolate(
                X_s2_norm, size=(hr_size, hr_size), mode="bilinear", align_corners=False
            )
            cond_s2 = model.model.first_stage_model.encode(X_s2_up).mode()
        else:
            cond_s2 = torch.nn.functional.interpolate(
                X_s2_norm, size=(latent_size, latent_size), mode="bilinear", align_corners=False
            )
        return cond_s2.to(model.device)
    return _encode


def main():
    parser = argparse.ArgumentParser(description="Inference with locally trained VAE+UNet")
    parser.add_argument("--unet_ckpt", type=str, default=str(DEFAULT_UNET_CKPT),
                        help="Path to LitUNetDenoiser Lightning checkpoint")
    parser.add_argument("--sampling_steps", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="CFG guidance scale (1.0 = disabled)") #use ~6.0 for best results
    parser.add_argument("--out_dir", type=str, default=str(TEST_DIR / "results"),
                        help="Parent directory for the output folder")
    parser.add_argument("--npz", type=str, default=str(TEST_DIR / "2025_1km_6096_725-ny.npz"),
                        help="Path to input NPZ tile")
    parser.add_argument("--config", type=str,
                        default=str(ROOT / "opensr_model" / "configs" / "config_10m.yaml"),
                        help="Model config YAML (use config_10m_no_s1.yaml for the S2-only model)")
    parser.add_argument("--include", choices=["all", "s2"], default="all",
                        help="'all' = S2+S1 conditioning; 's2' = true S2-only (4ch) for the no-S1 model")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None,
                        help="Device to run on (default: auto-detect, cuda if available)")
    args = parser.parse_args()

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    npz_file = pathlib.Path(args.npz)
    npz_stem = npz_file.stem
    ckpt_stem = pathlib.Path(args.unet_ckpt).stem
    m = re.search(r'epoch=(\d+).*val_loss=([\d.]+)', ckpt_stem)
    short_ckpt = f"e{int(m.group(1))}-val{m.group(2)}" if m else ckpt_stem
    include_tag = "" if args.include == "all" else "-kun-s2"
    run_name = f"{short_ckpt}_steps{args.sampling_steps}_gs{args.guidance_scale}_{npz_stem}{include_tag}"
    out_dir = pathlib.Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Build model
    cfg = OmegaConf.load(args.config)
    print(f"Building SRLatentDiffusion from {pathlib.Path(args.config).name}...")
    model = SRLatentDiffusion(cfg, device=device)

    # Load trained weights
    print(f"Loading checkpoint: {args.unet_ckpt}")
    load_trained_weights(model, args.unet_ckpt)

    # S2-only conditioning: swap in a true 4ch encoder (no S1 concat) for the no-S1 model.
    if args.include == "s2":
        model._tensor_encode = make_s2_only_encode(model)
        print("Conditioning: S2-only (4ch, no S1)")

    model.eval()

    # Load test tile (aerial optional — GT panel blank if absent)
    ds = FusionDataset(root=TEST_DIR, file_list=[str(npz_file)])
    sample = ds[0]
    s1 = sample["s1"].unsqueeze(0)    # (1, 2, 128, 128) zero-padded
    s2 = sample["s2"].unsqueeze(0)    # (1, 4, 128, 128) zero-padded
    aerial = sample["aerial"].unsqueeze(0)
    has_aerial = aerial.any().item()
    print(f"Input: s1={tuple(s1.shape)}, s2={tuple(s2.shape)}")

    # Sanity check: verify UNet produces non-degenerate output.
    # Conditioning channels = UNet in_channels - 4 noise channels
    # (6 for S2+S1 / in_channels=10; 4 for S2-only / in_channels=8).
    cond_ch = cfg.cond_stage_config.in_channels - 4
    with torch.no_grad():
        dummy_z = torch.randn(1, 4, 64, 64).to(device)
        dummy_t = torch.tensor([500]).to(device)
        dummy_cond = torch.randn(1, cond_ch, 64, 64).to(device)
        dummy_out = model.model.apply_model(dummy_z, dummy_t, cond=dummy_cond)
        print(f"UNet sanity check: mean={dummy_out.mean():.4f}, std={dummy_out.std():.4f}")

    # Display geometry based on model's scale_factor (ported from test-inference-batch.py).
    # sf<=2 (5m): pass pre-padded input so revert_padding is a no-op, then crop manually.
    # sf>2  (1m): pass native input; revert_padding (×4) strips the border.
    lr_pad = LR_PAD   # 14
    sf = model.scale_factor
    if sf <= 2:
        hr_out     = LR_PAD_SIZE * sf              # 256
        hr_crop    = lr_pad * sf                   # 28 — border in output space
        display_hr = hr_out - 2 * hr_crop          # 200
        hr_pad_gt  = (HR_PAD_SIZE - display_hr) // 2
        use_native = False
    else:
        display_hr = ORIG_HR                       # 1000
        hr_pad_gt  = HR_PAD
        use_native = True

    s2_native = s2[:, :, lr_pad:lr_pad + ORIG_LR, lr_pad:lr_pad + ORIG_LR]  # (1,4,100,100)
    s1_native = s1[:, :, lr_pad:lr_pad + ORIG_LR, lr_pad:lr_pad + ORIG_LR]  # (1,2,100,100)
    in_s2 = s2_native if use_native else s2
    in_s1 = s1_native if use_native else s1

    # Run inference
    print(f"Running DDIM sampling ({args.sampling_steps} steps)...")
    with torch.no_grad():
        sr = model.forward(in_s2.to(device), in_s1.to(device),
                           sampling_steps=args.sampling_steps,
                           guidance_scale=args.guidance_scale,
                           histogram_matching=False)
    print(f"Output SR: {tuple(sr.shape)}, min={sr.min().item():.1f}, max={sr.max().item():.1f}")

    # Crop SR / GT to display resolution
    if use_native:
        sr_crop = sr.cpu()
    else:
        sr_crop = sr[:, :, hr_crop:hr_crop + display_hr, hr_crop:hr_crop + display_hr].cpu()
    aerial_crop = aerial[:, :, hr_pad_gt:hr_pad_gt + display_hr, hr_pad_gt:hr_pad_gt + display_hr]

    # Upsample LR inputs + SR to display resolution for the side-by-side panel
    s2_up = F.interpolate(s2_native, size=(display_hr, display_hr), mode="bilinear", align_corners=False)
    s1_up = F.interpolate(s1_native, size=(display_hr, display_hr), mode="bilinear", align_corners=False)
    sr_up = F.interpolate(sr_crop,   size=(display_hr, display_hr), mode="bilinear", align_corners=False)

    s2_max = s2_up[:, :3].max().clamp(min=1e-6)
    s2_img = tensor_to_rgb((s2_up / s2_max * 255).clamp(0, 255))
    sr_img = tensor_to_rgb(sr_up)
    s1_img = tensor_to_s1_rgb(s1_up)
    if has_aerial:
        gt_img = tensor_to_rgb(aerial_crop)
    else:
        from PIL import ImageDraw
        _pil = Image.fromarray(np.full((display_hr, display_hr, 3), 128, dtype=np.uint8))
        ImageDraw.Draw(_pil).text((4, 4), "GT (N/A)", fill=(200, 200, 200))
        gt_img = np.array(_pil)

    comparison = np.concatenate([s1_img, s2_img, sr_img, gt_img], axis=1)
    out_path = out_dir / f"{npz_stem}.png"
    Image.fromarray(comparison).save(out_path)
    print(f"Saved {out_path}")
    print("Panel: S1 (VV/VH) | S2 input | SR prediction | Aerial GT")


if __name__ == "__main__":
    main()
