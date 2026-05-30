"""Cascade SR inference: S2+S1 (10m) → 5m aerial → 1m aerial.

Chains two trained UNets:
  Stage 1: S2+S1 (10m) → 5m aerial  (5m UNet, config_10m.yaml)
  Stage 2: 5m aerial + S1 → 1m aerial  (5m-to-1m UNet, config_1m.yaml)

Stage 2 conditioning is built manually (bypasses srmodel.forward()) because
srmodel._tensor_encode is designed for S2 input (normalize_s2), but stage 2
needs aerial normalization (normalize_aerial) for the 5m conditioning.

Usage:
    python test/test-inference-cascade.py \\
        --input_dir ~/npz/apr2025/5m-npz \\
        --unet_ckpt_5m checkpoints/5m/unet/last.ckpt \\
        --unet_ckpt_1m checkpoints/5to1m/unet/last.ckpt \\
        --npz_dir_1m ~/npz/apr2025/1m-npz

Output: side-by-side PNG per tile — S2 input | S1 | Stage1 5m SR | Stage2 1m SR | GT aerial
"""

import argparse
import pathlib
import re
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opensr_model.srmodel import SRLatentDiffusion
from opensr_model.utils import normalize_aerial, normalize_s1, normalize_s2

AERIAL_KEYS = ["aerial_r", "aerial_g", "aerial_b", "aerial_nir"]
S2_KEYS     = ["s2_r", "s2_g", "s2_b", "s2_nir"]
S1_KEYS     = ["s1_vv", "s1_vh"]

LR_PAD     = 128    # S1/S2 padded input size
LR_NATIVE  = 100    # S1/S2 native tile size
AERIAL_5M_PAD  = 256    # 5m aerial padded size (200px native)
AERIAL_1M_PAD  = 1024   # 1m aerial padded size
AERIAL_1M_NATIVE = 1000 # 1m aerial native size
LATENT_1M = 128         # 1024 / vae_downscale(8)


def _zero_pad(tensor: torch.Tensor, target: int) -> torch.Tensor:
    _, h, w = tensor.shape
    ph, pw = target - h, target - w
    if ph == 0 and pw == 0:
        return tensor
    return F.pad(tensor, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))


def load_trained_weights(model: SRLatentDiffusion, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    remapped = {k[len("ldm."):]: v for k, v in state_dict.items() if k.startswith("ldm.")}
    missing, unexpected = model.model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print(f"  Loaded {len(remapped)} keys from {ckpt_path}")


def tensor_to_rgb(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 4:
        t = t[0]
    img = t[:3].cpu().float().numpy()
    return np.transpose(np.clip(img, 0, 255).astype(np.uint8), (1, 2, 0))


def tensor_to_s1_rgb(t: torch.Tensor) -> np.ndarray:
    vv = t[0, 0].cpu().numpy()
    vh = t[0, 1].cpu().numpy()

    def stretch(a):
        lo, hi = np.percentile(a, 2), np.percentile(a, 98)
        return (np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)

    vv_n, vh_n = stretch(vv), stretch(vh)
    return np.stack([vv_n, vh_n, vv_n], axis=-1)


def blank_panel(size: int, label: str) -> np.ndarray:
    img = np.full((size, size, 3), 128, dtype=np.uint8)
    pil = Image.fromarray(img)
    ImageDraw.Draw(pil).text((4, 4), label, fill=(200, 200, 200))
    return np.array(pil)


@torch.no_grad()
def run_stage2(
    model2: SRLatentDiffusion,
    sr_5m: torch.Tensor,
    s1_128: torch.Tensor,
    s2_128: torch.Tensor,
    device: str,
    steps: int,
    guidance: float,
    cfg_plus_plus: bool = False,
) -> torch.Tensor:
    """Run 5m→1m inference without using model2.forward().

    model2.forward() assumes LR-sized X_s2 input (100×100 native) and applies
    size-dependent no_data_mask and revert_padding that break for 5m aerial input.
    This function replicates just the conditioning + DDIM loop.

    Args:
        sr_5m:   (1, 4, ~256, ~256)  stage1 output in aerial pixel space [0, 255]
        s1_128:  (1, 2, 128, 128)    S1 padded to LR_PAD
        s2_128:  (1, 4, 128, 128)    S2 padded to LR_PAD

    Returns:
        (1, 4, 1024, 1024) in [0, 255]
    """
    dtype = next(model2.model.first_stage_model.parameters()).dtype

    # Resize sr_5m to 256×256 — matches aerial_5m size in training NPZ files
    _, _, h, w = sr_5m.shape
    if h != AERIAL_5M_PAD or w != AERIAL_5M_PAD:
        sr_5m = F.interpolate(sr_5m, (AERIAL_5M_PAD, AERIAL_5M_PAD),
                              mode="bilinear", align_corners=False)

    # Encode 5m aerial through 1m VAE — unscaled, matches _build_conditioning in training
    a_norm = normalize_aerial(sr_5m.to(device), stage="norm").to(dtype)
    a_up   = F.interpolate(a_norm, (AERIAL_1M_PAD, AERIAL_1M_PAD),
                           mode="bilinear", align_corners=False)
    z_5m   = model2.model.first_stage_model.encode(a_up).mode()  # (1,4,128,128)

    # S1 conditioning: s1 at 128×128 → interpolate to latent size (no-op)
    s1_norm = normalize_s1(s1_128.to(device), stage="norm").to(dtype)
    s1_up   = F.interpolate(s1_norm, (LATENT_1M, LATENT_1M),
                            mode="bilinear", align_corners=False)

    # S2 conditioning: encode through 1m VAE — matches _build_conditioning in training
    s2_norm = normalize_s2(s2_128.to(device), stage="norm").to(dtype)
    s2_up   = F.interpolate(s2_norm, (AERIAL_1M_PAD, AERIAL_1M_PAD),
                            mode="bilinear", align_corners=False)
    z_s2    = model2.model.first_stage_model.encode(s2_up).mode()  # (1,4,128,128)

    conditioning      = torch.cat([z_5m, s1_up, z_s2], dim=1)   # (1, 10, 128, 128)
    null_conditioning = torch.zeros_like(conditioning)

    ddim, latent, time_range = model2._prepare_model(conditioning, custom_steps=steps)

    for i, step in enumerate(time_range):
        index = steps - i - 1
        t = torch.full((1,), step, device=device, dtype=torch.long)
        if cfg_plus_plus or guidance > 1.0:
            e_uncond = model2.model.apply_model(latent, t, cond=null_conditioning)
            e_cond   = model2.model.apply_model(latent, t, cond=conditioning)
            if cfg_plus_plus:
                latent = model2._ddim_step_cfg_pp(latent, e_cond, e_uncond, guidance, index, ddim, 1.0)
            else:
                e_t    = e_uncond + guidance * (e_cond - e_uncond)
                latent = model2._ddim_step(latent, e_t, index, ddim, 1.0)
        else:
            outs = ddim.p_sample_ddim(x=latent, c=conditioning, t=step, index=index,
                                      use_original_steps=False, temperature=1.0)
            latent, _ = outs

    return model2._tensor_decode(latent, spe_cor=False)  # (1, 4, 1024, 1024) in [0, 255]


def main():
    parser = argparse.ArgumentParser(description="Cascade SR inference: S2+S1 → 5m → 1m")
    parser.add_argument("--input_dir",    type=str, required=True,
                        help="Directory of 5m NPZ tiles (has S1, S2, and optionally 5m aerial)")
    parser.add_argument("--unet_ckpt_5m", type=str, default=None,
                        help="Stage 1 checkpoint (default: checkpoints/5m/unet/last.ckpt)")
    parser.add_argument("--unet_ckpt_1m", type=str, default=None,
                        help="Stage 2 checkpoint (default: checkpoints/5to1m/unet/last.ckpt)")
    parser.add_argument("--npz_dir_1m",   type=str, default=None,
                        help="Optional 1m NPZ directory for GT aerial ground-truth panel")
    parser.add_argument("--steps",        type=int,   default=100)
    parser.add_argument("--guidance",     type=float, default=6.0)
    parser.add_argument("--cfg_plus_plus", action="store_true", default=False,
                        help="Use CFG++ (x0-space guidance). Try lower scales (0.05–0.3).")
    parser.add_argument("--zero_s1", action="store_true", default=False,
                        help="Zero out S1 input (ablation: S2-only conditioning).")
    parser.add_argument("--zero_s2", action="store_true", default=False,
                        help="Zero out S2 input (ablation: S1-only conditioning).")
    parser.add_argument("--out_dir",      type=str,   default=str(ROOT / "test" / "results"))
    parser.add_argument("--device",       type=str,   default=None)
    parser.add_argument("--seed",         type=int,   default=42,
                        help="Random seed for deterministic generation")
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    if args.unet_ckpt_5m is None:
        args.unet_ckpt_5m = str(ROOT / "checkpoints" / "5m" / "unet" / "last.ckpt")
    if args.unet_ckpt_1m is None:
        args.unet_ckpt_1m = str(ROOT / "checkpoints" / "5to1m" / "unet" / "last.ckpt")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build output dir name from checkpoint stems
    stem1 = pathlib.Path(args.unet_ckpt_5m).stem
    stem2 = pathlib.Path(args.unet_ckpt_1m).stem
    m1    = re.search(r'epoch=(\d+)', stem1)
    m2    = re.search(r'epoch=(\d+)', stem2)
    label = (f"cascade_5m-e{int(m1.group(1)) if m1 else 'x'}"
             f"_1m-e{int(m2.group(1)) if m2 else 'x'}"
             f"_g{args.guidance:g}"
             f"{'_cpp' if args.cfg_plus_plus else ''}"
             f"{'_nos1' if args.zero_s1 else ''}"
             f"{'_nos2' if args.zero_s2 else ''}")
    run_dir = pathlib.Path(args.out_dir) / label
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}")

    # Load models
    cfg1 = OmegaConf.load(ROOT / "opensr_model" / "configs" / "config_10m.yaml")
    cfg2 = OmegaConf.load(ROOT / "opensr_model" / "configs" / "config_5m_to_1m_with_s2.yaml")

    print("Loading stage 1 (5m UNet, config_10m) ...")
    model1 = SRLatentDiffusion(cfg1, device=device)
    load_trained_weights(model1, args.unet_ckpt_5m)
    model1.eval()

    # Patch stage-1 conditioning so ablation flags zero the latent channels after
    # encoding rather than zeroing the raw pixels (which maps 0 DN → -1 in S2 and
    # 0 dB → +1 in S1, giving extreme out-of-distribution conditioning instead of null).
    if args.zero_s2 or args.zero_s1:
        _orig_encode = model1._tensor_encode
        if args.zero_s2 and args.zero_s1:
            def _patched_encode(X_s2, X_s1):
                cond = _orig_encode(X_s2, X_s1)
                cond[:] = 0
                return cond
        elif args.zero_s2:
            def _patched_encode(X_s2, X_s1):
                cond = _orig_encode(X_s2, X_s1)
                cond[:, :4] = 0   # zero S2 latent channels, keep S1
                return cond
        else:
            def _patched_encode(X_s2, X_s1):
                cond = _orig_encode(X_s2, X_s1)
                cond[:, 4:] = 0   # zero S1 channels, keep S2
                return cond
        model1._tensor_encode = _patched_encode
        ablated = ("S2" if args.zero_s2 else "") + ("S1" if args.zero_s1 else "")
        print(f"Stage-1 ablation: {ablated} conditioning zeroed after encoding")

    print("Loading stage 2 (5m→1m UNet, config_1m) ...")
    model2 = SRLatentDiffusion(cfg2, device=device)
    load_trained_weights(model2, args.unet_ckpt_1m)
    model2.eval()

    # Optional GT lookup from 1m NPZ dir
    gt_lookup: dict[str, pathlib.Path] = {}
    if args.npz_dir_1m:
        for p in pathlib.Path(args.npz_dir_1m).expanduser().glob("*.npz"):
            gt_lookup[p.stem] = p

    # Find input tiles
    input_dir  = pathlib.Path(args.input_dir).expanduser()
    tile_paths = sorted(input_dir.glob("*.npz"))
    if not tile_paths:
        print("No .npz tiles found. Exiting.")
        return
    print(f"Processing {len(tile_paths)} tiles...")

    lrp = (LR_PAD - LR_NATIVE) // 2          # 14 — LR border pixels to strip
    hrp = (AERIAL_1M_PAD - AERIAL_1M_NATIVE) // 2   # 12 — 1m HR border pixels

    for npz_path in tqdm(tile_paths, desc="Cascade"):
        stem = npz_path.stem

        # Load S2 and S1 from 5m NPZ
        with np.load(npz_path, allow_pickle=True) as f:
            s2_raw = torch.from_numpy(np.stack([f[k].astype(np.float32) for k in S2_KEYS]))
            s1_raw = torch.from_numpy(np.stack([f[k].astype(np.float32) for k in S1_KEYS]))

        s2_pad = _zero_pad(s2_raw, LR_PAD).unsqueeze(0)  # (1,4,128,128)
        s1_pad = _zero_pad(s1_raw, LR_PAD).unsqueeze(0)  # (1,2,128,128)

        # Display/stage-2 copies with visual zeroing for ablation panels
        s2_disp = torch.zeros_like(s2_pad) if args.zero_s2 else s2_pad
        s1_disp = torch.zeros_like(s1_pad) if args.zero_s1 else s1_pad

        # Native crop — used only for S2/S1 display panels
        s2_native = s2_disp[:, :, lrp:lrp + LR_NATIVE, lrp:lrp + LR_NATIVE]  # (1,4,100,100)
        s1_native = s1_disp[:, :, lrp:lrp + LR_NATIVE, lrp:lrp + LR_NATIVE]  # (1,2,100,100)

        # Stage 1: S2 + S1 → 5m aerial  (256×256)
        # Pass pre-padded 128×128: assert_tensor_validity is a no-op (padding=0) so
        # revert_padding strips nothing, giving the correct 256×256 output.
        # Passing native 100×100 would trigger the ×4 hardcoded strip in revert_padding,
        # which over-crops (144×144) and produces dark output for scale_factor=2.
        with torch.no_grad():
            sr_5m = model1.forward(
                s2_pad.to(device), s1_pad.to(device),  # real data; patch zeros conditioning
                sampling_steps=args.steps, guidance_scale=args.guidance,
                cfg_plus_plus=args.cfg_plus_plus,
                histogram_matching=False,
                apply_nodata_mask=False,
            ).cpu()  # (1, 4, 256, 256) — matches AERIAL_5M_PAD, run_stage2 skips resize

        # Stage 2: 5m aerial + S1 (padded) → 1m aerial  (1024×1024)
        # s1_disp (128×128) matches the training distribution of LatentFusion5mTo1mDataset
        sr_1m = run_stage2(model2, sr_5m, s1_disp, s2_pad, device, args.steps, args.guidance,
                           args.cfg_plus_plus).cpu()

        # Load GT 1m aerial if available
        gt_path = gt_lookup.get(stem)
        if gt_path:
            with np.load(gt_path, allow_pickle=True) as f:
                gt_raw = torch.from_numpy(np.stack([f[k].astype(np.float32) for k in AERIAL_KEYS]))
            gt_pad = _zero_pad(gt_raw, AERIAL_1M_PAD).unsqueeze(0)  # (1,4,1024,1024)
            has_gt = True
        else:
            has_gt = False

        # --- Build display panels at 1000×1000 ---
        display = AERIAL_1M_NATIVE  # 1000

        # S2: normalize and upsample
        s2_disp = F.interpolate(s2_native, (display, display), mode="bilinear", align_corners=False)
        s2_max  = s2_disp[:, :3].max().clamp(min=1e-6)
        s2_img  = tensor_to_rgb((s2_disp / s2_max * 255).clamp(0, 255))

        # S1
        s1_disp = F.interpolate(s1_native, (display, display), mode="bilinear", align_corners=False)
        s1_img  = tensor_to_s1_rgb(s1_disp)

        # Stage1 5m SR: crop native 200×200 content (strip 28px zero-masked border) then display
        _p5      = lrp * 2   # 14×2=28 — upscaled LR border at scale_factor=2
        sr_5m_native = sr_5m[:, :, _p5:AERIAL_5M_PAD - _p5, _p5:AERIAL_5M_PAD - _p5]
        stg1_up  = F.interpolate(sr_5m_native, (display, display), mode="bilinear", align_corners=False)
        stg1_img = tensor_to_rgb(stg1_up)

        # Stage2 1m SR cropped to native
        sr_crop  = sr_1m[:, :, hrp:hrp + display, hrp:hrp + display]
        stg2_img = tensor_to_rgb(sr_crop)

        # GT aerial
        if has_gt:
            gt_crop = gt_pad[:, :, hrp:hrp + display, hrp:hrp + display]
            gt_img  = tensor_to_rgb(gt_crop)
        else:
            gt_img  = blank_panel(display, "GT (N/A)")

        panel = np.concatenate([s2_img, s1_img, stg1_img, stg2_img, gt_img], axis=1)
        Image.fromarray(panel).save(run_dir / f"{stem}.png")

    print(f"\nDone! {len(tile_paths)} tiles saved to {run_dir}/")
    print("Each image: S2 input | S1 | Stage1 5m SR | Stage2 1m SR | GT aerial")


if __name__ == "__main__":
    main()
