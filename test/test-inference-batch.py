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
from opensr_model.data import FusionDataset, LR_PAD_SIZE, HR_PAD_SIZE, LR_NATIVE, HR_NATIVE, LR_PAD, HR_PAD

DEFAULT_UNET_CKPT = ROOT / "checkpoints" / "1m" / "unet" / "last.ckpt"
DEFAULT_INPUT_DIR = pathlib.Path("~/npz/apr2025/1m-untouched").expanduser()

ORIG_LR = LR_NATIVE   # 100
ORIG_HR = HR_NATIVE   # 1000


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
    parser.add_argument("--cfg_plus_plus", action="store_true", default=False,
                        help="Use CFG++ (x0-space guidance). Try lower scales (0.05–0.3) "
                             "— less artifacts than standard CFG at equivalent strength.")
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
    parser.add_argument("--config", type=str, default=None,
                        help="Config yaml path (default: config_opensr.yaml for --opensr, "
                             "config_1m.yaml otherwise). Use config_10m.yaml for the 5m model.")
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
        ckpt_stem = pathlib.Path(args.unet_ckpt).stem
        m = re.search(r'epoch=(\d+)', ckpt_stem)
        g_str = f"g{args.guidance:g}{'pp' if args.cfg_plus_plus else ''}"
        res_tag = "5m" if args.config and "10m" in args.config else "1m"
        ckpt_label = f"{res_tag}-e{int(m.group(1))}-{g_str}" if m else f"{res_tag}-{ckpt_stem}-{g_str}"
        run_dir = pathlib.Path(args.out_dir) / ckpt_label
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_dir}")

    # Build model
    if args.config:
        cfg_path = pathlib.Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = ROOT / cfg_path
    else:
        cfg_path = ROOT / "opensr_model" / "configs" / (
            "config_opensr.yaml" if args.opensr else "config_1m.yaml"
        )
    cfg = OmegaConf.load(cfg_path)
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

    # Determine display geometry based on model's scale_factor.
    # revert_padding() in utils.py hard-codes ×4; correct only for scale_factor≥8.
    # For scale_factor=2 (5m model) we pass the pre-padded 128×128 input so that
    # assert_tensor_validity adds no extra padding (padding=0) and revert_padding
    # becomes a no-op, then we crop the native content manually.
    lr_pad = LR_PAD   # 14
    sf = model.scale_factor  # 2 for config_10m (5m), 8 for config_1m (1m)
    if sf <= 2:
        # 5m model: pass pre-padded input, crop output manually
        hr_out     = LR_PAD_SIZE * sf           # 256
        hr_crop    = lr_pad * sf                # 28 — border in output space
        display_hr = hr_out - 2 * hr_crop       # 200
        hr_pad_gt  = (HR_PAD_SIZE - display_hr) // 2  # 412 — where aerial sits in 1024-pad
        use_native = False
    else:
        # 1m model: native input, revert_padding (×4) strips 56px → 912×912
        display_hr = ORIG_HR   # 1000
        hr_pad_gt  = HR_PAD    # 12
        use_native = True

    # Load all tiles (aerial optional — GT panel will be blank if absent)
    ds = FusionDataset(args.input_dir, require_aerial=False, pad=True)
    if len(ds) == 0:
        print("No valid tiles found. Exiting.")
        return
    print(f"Processing {len(ds)} tiles (scale_factor={sf}, display={display_hr}×{display_hr})...")

    for i in tqdm(range(len(ds)), desc="Inference"):
        sample = ds[i]
        name = pathlib.Path(sample["path"]).stem

        s1 = sample["s1"].unsqueeze(0)   # (1, 2, 128, 128) zero-padded
        s2 = sample["s2"].unsqueeze(0)   # (1, 4, 128, 128) zero-padded
        aerial = sample["aerial"].unsqueeze(0)
        has_aerial = aerial.any().item()

        s2_native = s2[:, :, lr_pad:lr_pad + ORIG_LR, lr_pad:lr_pad + ORIG_LR]  # (1,4,100,100)
        s1_native = s1[:, :, lr_pad:lr_pad + ORIG_LR, lr_pad:lr_pad + ORIG_LR]  # (1,2,100,100)

        # For scale_factor=2: pass pre-padded 128×128 so assert_tensor_validity is a no-op
        # (padding=0 → revert_padding strips nothing → output is hr_out×hr_out).
        # For scale_factor≥8: pass native 100×100; assert_tensor_validity reflect-pads to 128×128.
        in_s2 = s2_native if use_native else s2
        in_s1 = s1_native if use_native else s1

        with torch.no_grad():
            sr: torch.Tensor = model.forward(
                in_s2.to(device), in_s1.to(device),
                sampling_steps=args.steps,
                guidance_scale=args.guidance,
                histogram_matching=False,
                cfg_plus_plus=args.cfg_plus_plus,
            )

        # Crop display inputs back to native
        s2_crop = s2_native
        if args.opensr:
            # 4× model: output is 400×400 (revert_padding already applied correctly)
            display_hr = ORIG_LR * 4          # 400
            sr_crop = sr.cpu()
        else:
            s1_crop = s1_native
            if use_native:
                sr_crop = sr.cpu()            # 912×912, no further crop
            else:
                sr_crop = sr[:, :, hr_crop:hr_crop + display_hr,
                                   hr_crop:hr_crop + display_hr].cpu()  # e.g. 200×200
            aerial_crop = aerial[:, :, hr_pad_gt:hr_pad_gt + display_hr,
                                       hr_pad_gt:hr_pad_gt + display_hr]

        # Upsample LR inputs and SR to display resolution for side-by-side comparison
        s2_up = F.interpolate(s2_crop, size=(display_hr, display_hr), mode="bilinear", align_corners=False)
        if not args.opensr:
            s1_up = F.interpolate(s1_crop, size=(display_hr, display_hr), mode="bilinear", align_corners=False)
            sr_up = F.interpolate(sr_crop, size=(display_hr, display_hr), mode="bilinear", align_corners=False)

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
            sr_img = tensor_to_rgb(sr_up)

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
