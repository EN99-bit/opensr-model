"""Batch inference on a directory of NPZ tiles.

Loads a trained UNet checkpoint, runs super-resolution on every NPZ file in
input_dir, and saves a side-by-side comparison PNG per tile (all in one folder):

    S1 (VV/VH) | S2 input (upsampled) | SR prediction | Aerial ground truth

Usage:
    python test/test-inference-batch.py
    python test/test-inference-batch.py --input_dir ~/npz/apr2025/5m-untouched
    python test/test-inference-batch.py --steps 50 --guidance 4.0
    python test/test-inference-batch.py --unet_ckpt checkpoints/unet/best.ckpt
    python test/test-inference-batch.py --include s2
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
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opensr_model.srmodel import SRLatentDiffusion
from opensr_model.data import FusionDataset, LR_PAD_SIZE, HR_PAD_SIZE

DEFAULT_UNET_CKPT = ROOT / "checkpoints" / "unet" / "last.ckpt"
DEFAULT_INPUT_DIR = pathlib.Path("~/npz/apr2025/5m-untouched").expanduser()

# Native (unpadded) tile sizes
ORIG_LR = 100   # S1/S2 native pixels
ORIG_HR = 200   # Aerial native pixels


def load_trained_weights(model: SRLatentDiffusion, unet_ckpt: str):
    """Load UNet weights from a LitUNetDenoiser Lightning checkpoint."""
    ckpt = torch.load(unet_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    remapped = {k[len("ldm."):]: v for k, v in state_dict.items() if k.startswith("ldm.")}
    missing, unexpected = model.model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print(f"  Loaded {len(remapped)} keys from {unet_ckpt}")


def tensor_to_rgb(t):
    """Convert (1, C, H, W) or (C, H, W) tensor to (H, W, 3) uint8 numpy array (bands 0,1,2)."""
    if t.dim() == 4:
        t = t[0]
    img = t[:3, :, :].cpu().numpy()
    img = np.clip(img, 0, 255).astype(np.uint8)
    return np.transpose(img, (1, 2, 0))


def tensor_to_s1_rgb(t):
    """Convert S1 (1, 2, H, W) dB tensor to (H, W, 3) uint8 via percentile stretch.

    Displayed as R=VV, G=VH, B=VV.
    """
    vv = t[0, 0].cpu().numpy()
    vh = t[0, 1].cpu().numpy()

    def percentile_stretch(arr):
        lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
        arr = (arr - lo) / (hi - lo + 1e-6)
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)

    vv_n = percentile_stretch(vv)
    vh_n = percentile_stretch(vh)
    return np.stack([vv_n, vh_n, vv_n], axis=-1)


def main():
    parser = argparse.ArgumentParser(description="Batch SR inference on NPZ tiles")
    parser.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR),
                        help="Directory containing .npz tiles")
    parser.add_argument("--steps", type=int, default=100,
                        help="DDIM sampling steps")
    parser.add_argument("--guidance", type=float, default=6.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--unet_ckpt", type=str, default=None,
                        help="Path to LitUNetDenoiser Lightning checkpoint")
    parser.add_argument("--out_dir", type=str, default=str(ROOT / "test" / "results"),
                        help="Parent directory for output folder")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on: 'cuda' or 'cpu' (default: auto-detect)")
    parser.add_argument("--include", type=str, default="all", choices=["all", "s1", "s2"],
                        help="Which modalities to use: 'all' (S1+S2), 's1' only, 's2' only (default: all)")
    parser.add_argument("--opensr", action="store_true",
                        help="Use official OpenSR pretrained weights (S2-only, 8-ch UNet); "
                             "downloads from HuggingFace on first run")
    args = parser.parse_args()

    if args.opensr and args.unet_ckpt is not None:
        parser.error("--opensr and --unet_ckpt are mutually exclusive")
    if not args.opensr and args.unet_ckpt is None:
        args.unet_ckpt = str(DEFAULT_UNET_CKPT)

    if args.opensr and args.guidance != 1.0:
        print(f"  --opensr: overriding guidance_scale {args.guidance} → 1.0 "
              "(official model was not trained with CFG)")
        args.guidance = 1.0

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.opensr:
        run_dir = pathlib.Path(args.out_dir) / f"opensr_steps{args.steps}_gs{args.guidance}"
    else:
        # Extract epoch label from checkpoint filename (e.g. epoch=10-val_loss=0.279687 -> e10-val0.279687)
        ckpt_stem = pathlib.Path(args.unet_ckpt).stem
        m = re.search(r'epoch=(\d+).*val_loss=([\d.]+)', ckpt_stem)
        ckpt_label = f"e{int(m.group(1))}-val{m.group(2)}" if m else ckpt_stem
        modality_suffix = f"_{args.include}only" if args.include != "all" else ""
        run_dir = pathlib.Path(args.out_dir) / f"{ckpt_label}_batch_steps{args.steps}_gs{args.guidance}{modality_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_dir}")

    # Build model
    if args.opensr:
        cfg = OmegaConf.load(ROOT / "opensr_model" / "configs" / "config_opensr.yaml")
    else:
        cfg = OmegaConf.load(ROOT / "opensr_model" / "configs" / "config_10m.yaml")
    print("Building SRLatentDiffusion...")
    model = SRLatentDiffusion(cfg, device=device)

    if args.opensr:
        print(f"Loading official OpenSR weights from HuggingFace ({cfg.ckpt_version})...")
        model.load_pretrained(cfg.ckpt_version)
        model.scale_factor = 4  # official model is 4×; config_opensr.yaml defaults to 2
        def _s2_only_encode(X_s2, X_s1):
            model._X_s2 = X_s2.clone()
            lr_size = X_s2.shape[-1]
            hr_size = lr_size * model.scale_factor
            # Official model expects [0,1] reflectance; FusionDataset stores raw DN (0-10000)
            X_s2_refl = (X_s2 / 10000.0).clamp(0, 1)
            X_s2_up = F.interpolate(X_s2_refl, size=(hr_size, hr_size),
                                    mode="bilinear", align_corners=False)
            return model.model.first_stage_model.encode(X_s2_up).sample().to(model.device)
        def _s2_only_decode(latent, spe_cor=False):
            # Official model trained with scale_factor=1.0; srmodel.py hardcodes 0.18215
            # which would multiply the latent by 5.48× before decoding — saturating to white.
            # Call first_stage_model.decode() directly to bypass that erroneous scaling.
            decoded = model.model.first_stage_model.decode(latent)
            return ((decoded + 1.0) / 2.0).clamp(0, 1) * 255.0
        model._tensor_encode = _s2_only_encode
        model._tensor_decode = _s2_only_decode
    else:
        print(f"Loading checkpoint: {args.unet_ckpt}")
        load_trained_weights(model, args.unet_ckpt)
        if args.include != "all":
            _original_encode = model._tensor_encode
            if args.include == "s1":
                def _patched_encode(X_s2, X_s1):
                    cond = _original_encode(X_s2, X_s1)
                    cond[:, :4] = 0
                    return cond
            else:
                def _patched_encode(X_s2, X_s1):
                    cond = _original_encode(X_s2, X_s1)
                    cond[:, 4:] = 0
                    return cond
            model._tensor_encode = _patched_encode
            print(f"Ablation: using {args.include} only (other modality zeroed in conditioning)")
    model.eval()

    # Load all tiles (aerial optional — GT panel will be blank if absent)
    ds = FusionDataset(args.input_dir, require_aerial=False, pad=True)
    if len(ds) == 0:
        print("No valid tiles found. Exiting.")
        return
    print(f"Processing {len(ds)} tiles...")

    lr_pad = (LR_PAD_SIZE - ORIG_LR) // 2
    hr_pad = (HR_PAD_SIZE - ORIG_HR) // 2

    for i in tqdm(range(len(ds)), desc="Inference"):
        sample = ds[i]
        name = pathlib.Path(sample["path"]).stem

        s1 = sample["s1"].unsqueeze(0)   # (1, 2, 128, 128)
        s2 = sample["s2"].unsqueeze(0)   # (1, 4, 128, 128)
        aerial = sample["aerial"].unsqueeze(0)  # (1, 4, 256, 256) — zeros if no aerial in tile
        has_aerial = aerial.any().item()

        with torch.no_grad():
            sr: torch.Tensor = model.forward(
                s2.to(device), s1.to(device),
                sampling_steps=args.steps,
                guidance_scale=args.guidance,
                histogram_matching=False,
            )  # (1, 4, 256, 256), values 0-255

        # Crop padding back to native sizes
        s2_crop = s2[:, :, lr_pad:lr_pad + ORIG_LR, lr_pad:lr_pad + ORIG_LR]
        if args.opensr:
            # 4× model: output is 512×512, native content is 400×400 offset by the upscaled pad
            display_hr = ORIG_LR * 4          # 400
            _pad = lr_pad * 4                 # 14 * 4 = 56 — zero-pad border at 4× scale
            sr_crop = sr[:, :, _pad:_pad + display_hr, _pad:_pad + display_hr].cpu()
        else:
            display_hr = ORIG_HR
            s1_crop = s1[:, :, lr_pad:lr_pad + ORIG_LR, lr_pad:lr_pad + ORIG_LR]
            sr_crop = sr[:, :, hr_pad:hr_pad + ORIG_HR, hr_pad:hr_pad + ORIG_HR].cpu()
            aerial_crop = aerial[:, :, hr_pad:hr_pad + ORIG_HR, hr_pad:hr_pad + ORIG_HR]

        # Upsample S2 to display resolution for side-by-side comparison
        s2_up = F.interpolate(s2_crop, size=(display_hr, display_hr), mode="bilinear", align_corners=False)
        if not args.opensr:
            s1_up = F.interpolate(s1_crop, size=(display_hr, display_hr), mode="bilinear", align_corners=False)

        # Build panel images
        if args.opensr:
            # Percentile stretch for both S2 and SR so contrast matches
            s2_np = s2_up[0, :3].cpu().numpy()
            s2_lo, s2_hi = np.percentile(s2_np, 2), np.percentile(s2_np, 98)
            s2_img = tensor_to_rgb(((s2_up - s2_lo) / (s2_hi - s2_lo + 1e-6)).clamp(0, 1) * 255)
            sr_np = sr_crop[0, :3].cpu().numpy()
            sr_lo, sr_hi = np.percentile(sr_np, 2), np.percentile(sr_np, 98)
            sr_img = tensor_to_rgb(((sr_crop - sr_lo) / (sr_hi - sr_lo + 1e-6)).clamp(0, 1) * 255)
        else:
            s2_max = s2_up[:, :3].max().clamp(min=1e-6)
            s2_rgb = (s2_up / s2_max * 255).clamp(0, 255)
            s2_img = tensor_to_rgb(s2_rgb)
            sr_img = tensor_to_rgb(sr_crop)

        if args.opensr:
            from PIL import ImageDraw
            def _blank(label):
                img = np.full((display_hr, display_hr, 3), 128, dtype=np.uint8)
                pil = Image.fromarray(img)
                ImageDraw.Draw(pil).text((4, 4), label, fill=(200, 200, 200))
                return np.array(pil)
            s1_img  = _blank("S1 (not used)")
            gt_img  = _blank("GT (N/A)")
        else:
            s1_img = tensor_to_s1_rgb(s1_up)
            if has_aerial:
                gt_img = tensor_to_rgb(aerial_crop)
            else:
                from PIL import ImageDraw
                _blank_arr = np.full((display_hr, display_hr, 3), 128, dtype=np.uint8)
                _pil = Image.fromarray(_blank_arr)
                ImageDraw.Draw(_pil).text((4, 4), "GT (N/A)", fill=(200, 200, 200))
                gt_img = np.array(_pil)

        comparison = np.concatenate([s1_img, s2_img, sr_img, gt_img], axis=1)
        Image.fromarray(comparison).save(run_dir / f"{name}.png")

    print(f"\nDone! {len(ds)} images saved to {run_dir}/")
    print("Each image shows: S1 (VV/VH) | S2 input | SR prediction | Aerial GT")


if __name__ == "__main__":
    main()
