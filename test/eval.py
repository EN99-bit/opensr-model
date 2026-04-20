"""Evaluation script: compute opensr-test + PSNR + SSIM metrics on NPZ tiles.

Runs inference with a trained checkpoint and evaluates SR quality against aerial
ground truth, reporting the same metrics as the opensr-test benchmark paper:

    Reflectance | Spectral | Spatial | Synthesis |
    Hallucination | Omission | Improvement | PSNR | SSIM

Requires opensr-test: pip install opensr-test

Usage:
    python test/eval.py --npz_dir ~/npz/apr2025/5m-untouched
    python test/eval.py --npz_dir ~/npz/apr2025/5m-untouched --opensr
    python test/eval.py --npz_dir ~/npz/apr2025/5m-untouched --out_csv results.csv
    python test/eval.py --npz_dir ~/npz/apr2025/5m-untouched --include s2
"""

import argparse
import csv
import pathlib
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tqdm import tqdm

try:
    import opensr_test
except ImportError:
    print("opensr-test not installed. Run: pip install opensr-test")
    sys.exit(1)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opensr_model.srmodel import SRLatentDiffusion
from opensr_model.data import FusionDataset, LR_PAD_SIZE, HR_PAD_SIZE
from opensr_model.utils import normalize_s2

DEFAULT_UNET_CKPT = ROOT / "checkpoints" / "unet" / "last.ckpt"
DEFAULT_INPUT_DIR = pathlib.Path("~/npz/apr2025/5m-untouched").expanduser()

ORIG_LR = 100
ORIG_HR = 200

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


def tensor_to_rgb(t):
    if t.dim() == 4:
        t = t[0]
    img = t[:3, :, :].cpu().numpy()
    img = np.clip(img, 0, 255).astype(np.uint8)
    return np.transpose(img, (1, 2, 0))


def tensor_to_s1_rgb(t):
    vv = t[0, 0].cpu().numpy()
    vh = t[0, 1].cpu().numpy()

    def percentile_stretch(arr):
        lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
        arr = (arr - lo) / (hi - lo + 1e-6)
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)

    vv_n = percentile_stretch(vv)
    vh_n = percentile_stretch(vh)
    return np.stack([vv_n, vh_n, vv_n], axis=-1)


def load_trained_weights(model: SRLatentDiffusion, unet_ckpt: str):
    ckpt = torch.load(unet_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    remapped = {k[len("ldm."):]: v for k, v in state_dict.items() if k.startswith("ldm.")}
    missing, unexpected = model.model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:3]}{'...' if len(missing) > 3 else ''}")
    print(f"  Loaded {len(remapped)} keys from {unet_ckpt}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SR quality on NPZ tiles")
    parser.add_argument("--npz_dir", type=str, default=str(DEFAULT_INPUT_DIR),
                        help="Directory containing .npz tiles")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to trained checkpoint")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--include", type=str, default="all", choices=["all", "s1", "s2"],
                        help="Modalities to use: all, s1 only, s2 only")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_csv", type=str, default=None,
                        help="Optional path to save per-tile CSV results")
    parser.add_argument("--out_dir", type=str, default=str(ROOT / "test" / "results"),
                        help="Parent directory for output images")
    parser.add_argument("--opensr", action="store_true",
                        help="Use the official OpenSR pretrained model (S2-only, 8-ch UNet); "
                             "downloads from HuggingFace on first run")
    args = parser.parse_args()

    if args.opensr and args.ckpt is not None:
        parser.error("--opensr and --ckpt are mutually exclusive")
    if not args.opensr and args.ckpt is None:
        args.ckpt = str(DEFAULT_UNET_CKPT)

    if args.opensr and args.guidance != 1.0:
        print(f"  --opensr: overriding guidance_scale {args.guidance} → 1.0 "
              "(official model was not trained with CFG)")
        args.guidance = 1.0

    if args.opensr:
        run_dir = pathlib.Path(args.out_dir) / f"opensr_eval_steps{args.steps}_gs{args.guidance}"
    else:
        ckpt_stem = pathlib.Path(args.ckpt).stem
        m = re.search(r'epoch=(\d+).*val_loss=([\d.]+)', ckpt_stem)
        ckpt_label = f"e{int(m.group(1))}-val{m.group(2)}" if m else ckpt_stem
        modality_suffix = f"_{args.include}only" if args.include != "all" else ""
        run_dir = pathlib.Path(args.out_dir) / f"{ckpt_label}_eval_steps{args.steps}_gs{args.guidance}{modality_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_dir}")

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    if args.opensr:
        cfg = OmegaConf.load(ROOT / "opensr_model" / "configs" / "config_opensr.yaml")
    else:
        cfg = OmegaConf.load(ROOT / "opensr_model" / "configs" / "config_10m.yaml")
    model = SRLatentDiffusion(cfg, device=device)

    if args.opensr:
        print(f"Loading official OpenSR weights from HuggingFace ({cfg.ckpt_version})...")
        model.load_pretrained(cfg.ckpt_version)
        model.scale_factor = 4  # official model is 4×; config_opensr.yaml defaults to 2
        # Official model uses S2-only conditioning (4ch latent); skip S1
        def _s2_only_encode(X_s2, X_s1):
            model._X_s2 = X_s2.clone()
            lr_size = X_s2.shape[-1]
            hr_size = lr_size * model.scale_factor
            # Official model expects [0,1] reflectance; FusionDataset stores raw DN (0-10000)
            X_s2_refl = (X_s2 / 10000.0).clamp(0, 1)
            X_s2_up = torch.nn.functional.interpolate(
                X_s2_refl, size=(hr_size, hr_size), mode="bilinear", align_corners=False
            )
            cond_s2 = model.model.first_stage_model.encode(X_s2_up).sample()
            return cond_s2.to(model.device)
        def _s2_only_decode(latent, spe_cor=False):
            # Official model trained with scale_factor=1.0; bypass decode_first_stage
            # to avoid the erroneous 1/0.18215 ≈ 5.48× scaling in srmodel.py.
            decoded = model.model.first_stage_model.decode(latent)
            return ((decoded + 1.0) / 2.0).clamp(0, 1) * 255.0
        model._tensor_encode = _s2_only_encode
        model._tensor_decode = _s2_only_decode
    else:
        print(f"Loading checkpoint: {args.ckpt}")
        load_trained_weights(model, args.ckpt)
    model.eval()

    # Ablation: zero out S1 or S2 conditioning channels
    if args.include != "all":
        _orig = model._tensor_encode
        if args.include == "s1":
            def _patched(X_s2, X_s1):
                cond = _orig(X_s2, X_s1)
                cond[:, :4] = 0  # zero S2, keep S1
                return cond
        else:
            def _patched(X_s2, X_s1):
                cond = _orig(X_s2, X_s1)
                cond[:, 4:] = 0  # zero S1, keep S2
                return cond
        model._tensor_encode = _patched
        print(f"Ablation: {args.include} only")

    # Load tiles
    ds = FusionDataset(args.npz_dir, require_aerial=True, pad=True)
    if len(ds) == 0:
        print("No valid tiles found.")
        return
    print(f"Evaluating {len(ds)} tiles...")

    lr_pad = (LR_PAD_SIZE - ORIG_LR) // 2
    hr_pad = (HR_PAD_SIZE - ORIG_HR) // 2

    rows = []  # per-tile results

    for i in tqdm(range(len(ds)), desc="Eval"):
        sample = ds[i]
        name = pathlib.Path(sample["path"]).stem

        s2     = sample["s2"].unsqueeze(0)      # (1, 4, 128, 128)
        s1     = sample["s1"].unsqueeze(0)      # (1, 2, 128, 128)
        aerial = sample["aerial"].unsqueeze(0)  # (1, 4, 256, 256)

        with torch.no_grad():
            sr: torch.Tensor = model.forward(
                s2.to(device), s1.to(device),
                sampling_steps=args.steps,
                guidance_scale=args.guidance,
                histogram_matching=False,
            )

        # Crop padding to native sizes
        s2_crop = s2[:, :, lr_pad:lr_pad + ORIG_LR, lr_pad:lr_pad + ORIG_LR]
        s1_crop = s1[:, :, lr_pad:lr_pad + ORIG_LR, lr_pad:lr_pad + ORIG_LR]
        if args.opensr:
            _sr_pad = lr_pad * 4  # 14 * 4 = 56 — native content offset at 4× scale
            sr_crop = sr[:, :, _sr_pad:_sr_pad + ORIG_HR, _sr_pad:_sr_pad + ORIG_HR].cpu()
        else:
            sr_crop = sr[:, :, hr_pad:hr_pad + ORIG_HR, hr_pad:hr_pad + ORIG_HR].cpu()
        aerial_crop = aerial[:, :, hr_pad:hr_pad + ORIG_HR, hr_pad:hr_pad + ORIG_HR]

        # Normalize to [0, 1] for opensr_test
        # LR: S2 DN → [-1,1] via normalize_s2 → [0,1]
        lr_norm = (normalize_s2(s2_crop[0], stage="norm") + 1.0) / 2.0
        # SR: model output is [0, 255]
        sr_norm = (sr_crop[0] / 255.0).clamp(0, 1)
        # HR: aerial is [0, 255] uint8-origin
        hr_norm = (aerial_crop[0] / 255.0).clamp(0, 1)

        # opensr-test metrics
        try:
            metrics = opensr_test.Metrics()
            result = metrics.compute(lr=lr_norm, sr=sr_norm, hr=hr_norm)
        except Exception as e:
            print(f"  [{name}] opensr_test failed: {e}")
            result = {k: float("nan") for k in METRIC_KEYS[:7]}

        # PSNR + SSIM (RGB uint8)
        sr_u8 = (sr_norm[:3].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        hr_u8 = (hr_norm[:3].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        psnr = peak_signal_noise_ratio(hr_u8, sr_u8, data_range=255)
        ssim = structural_similarity(hr_u8, sr_u8, channel_axis=2, data_range=255)

        row = {"tile": name}
        for k in METRIC_KEYS[:7]:
            row[k] = float(result.get(k, float("nan")))
        row["psnr"] = float(psnr)
        row["ssim"] = float(ssim)
        rows.append(row)

        # Save comparison image: S1 | S2 | SR | GT
        s2_up = F.interpolate(s2_crop, size=(ORIG_HR, ORIG_HR), mode="bilinear", align_corners=False)
        s1_up = F.interpolate(s1_crop, size=(ORIG_HR, ORIG_HR), mode="bilinear", align_corners=False)
        if args.opensr:
            s2_np = s2_up[0, :3].cpu().numpy()
            s2_lo, s2_hi = np.percentile(s2_np, 2), np.percentile(s2_np, 98)
            s2_img = tensor_to_rgb(((s2_up - s2_lo) / (s2_hi - s2_lo + 1e-6)).clamp(0, 1) * 255)
            sr_np = sr_crop[0, :3].cpu().numpy()
            sr_lo, sr_hi = np.percentile(sr_np, 2), np.percentile(sr_np, 98)
            sr_img = tensor_to_rgb(((sr_crop - sr_lo) / (sr_hi - sr_lo + 1e-6)).clamp(0, 1) * 255)
        else:
            s2_max = s2_up[:, :3].max().clamp(min=1e-6)
            s2_img = tensor_to_rgb((s2_up / s2_max * 255).clamp(0, 255))
            sr_img = tensor_to_rgb(sr_crop)
        s1_img = tensor_to_s1_rgb(s1_up)
        gt_img = tensor_to_rgb(aerial_crop)
        comparison = np.concatenate([s1_img, s2_img, sr_img, gt_img], axis=1)
        Image.fromarray(comparison).save(run_dir / f"{name}.png")

    # Aggregate
    print("\n" + "=" * 52)
    print(f"  Results over {len(rows)} tiles")
    print("=" * 52)
    for k in METRIC_KEYS:
        vals = np.array([r[k] for r in rows], dtype=float)
        mean = np.nanmean(vals)
        std  = np.nanstd(vals)
        print(f"  {METRIC_LABELS[k]:<22}  {mean:8.4f} ± {std:.4f}")
    print("=" * 52)

    # Save CSV
    out = pathlib.Path(args.out_csv) if args.out_csv else run_dir / "metrics.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tile"] + METRIC_KEYS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-tile results saved to {out}")


if __name__ == "__main__":
    main()
