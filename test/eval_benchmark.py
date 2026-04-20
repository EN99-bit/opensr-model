"""Evaluate SR models against opensr-test benchmark datasets.

Downloads each dataset automatically from HuggingFace on first run (~100 MB each).

Two modes:
  default (no --unet_ckpt): official OpenSR pretrained weights (S2-only, 4× SR:
      121px LR → 484px HR). guidance defaults to 1.0.
  --unet_ckpt PATH: locally trained checkpoint (S2+S1, 2× SR: 121px LR → 242px HR).
      HR ground truth is downsampled 484 → 242 for comparison. guidance defaults to 6.0.

Usage:
    python test/eval_benchmark.py                              # all 6 datasets, official weights
    python test/eval_benchmark.py --dataset naip               # single dataset
    python test/eval_benchmark.py --dataset naip spot          # multiple datasets
    python test/eval_benchmark.py --out_csv benchmark.csv      # save per-sample CSV
    python test/eval_benchmark.py --unet_ckpt checkpoints/unet/last.ckpt
    python test/eval_benchmark.py --unet_ckpt checkpoints/unet/last.ckpt --dataset naip
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

try:
    import opensr_test
except ImportError:
    print("opensr-test not installed. Run: pip install opensr-test")
    sys.exit(1)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opensr_model.srmodel import SRLatentDiffusion

DEFAULT_UNET_CKPT = ROOT / "checkpoints" / "unet" / "last.ckpt"

ALL_DATASETS = ["naip", "spot", "venus", "spain_urban", "spain_crops", "satellogic"]

# L2A band indices for RGBNIR (B4/R=3, B3/G=2, B2/B=1, B8/NIR=7)
L2A_RGBNIR = [3, 2, 1, 7]

LR_NATIVE = 121  # S2 tile size in opensr-test
LR_PAD    = 128  # model input (pad right/bottom to reach this)
HR_NATIVE = 484  # HR ground-truth size (= LR_NATIVE * 4)

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


def pad_right_bottom(x: torch.Tensor, target: int) -> torch.Tensor:
    """Pad (B, C, H, W) to target×target by adding zeros on right and bottom."""
    _, _, h, w = x.shape
    return F.pad(x, (0, target - w, 0, target - h))


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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SR models on opensr-test benchmark datasets"
    )
    parser.add_argument("--unet_ckpt", type=str, default=None,
                        help="Path to LitUNetDenoiser Lightning checkpoint (2× model); "
                             "if omitted, uses official OpenSR pretrained weights (4× model)")
    parser.add_argument("--dataset", nargs="+", default=ALL_DATASETS,
                        choices=ALL_DATASETS, metavar="DATASET",
                        help="Dataset(s) to evaluate (default: all six)")
    parser.add_argument("--steps",    type=int,   default=100)
    parser.add_argument("--guidance", type=float, default=None,
                        help="CFG guidance scale (default: 1.0 for official model, 6.0 for custom)")
    parser.add_argument("--device",   type=str,   default=None)
    parser.add_argument("--out_csv",  type=str,   default=None,
                        help="Optional path to save per-sample CSV results")
    args = parser.parse_args()

    use_custom = args.unet_ckpt is not None
    if args.guidance is None:
        args.guidance = 6.0 if use_custom else 1.0

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if use_custom:
        # User's 2× model: S2+S1 conditioning, raw DN input, 2× scale
        cfg = OmegaConf.load(ROOT / "opensr_model" / "configs" / "config_10m.yaml")
        model = SRLatentDiffusion(cfg, device=device)
        print(f"Loading checkpoint: {args.unet_ckpt}")
        load_trained_weights(model, args.unet_ckpt)
        hr_eval_size = LR_NATIVE * 2  # 242 — 2× SR output size
    else:
        # Official OpenSR 4× model: S2-only conditioning, [0,1] reflectance input
        cfg = OmegaConf.load(ROOT / "opensr_model" / "configs" / "config_opensr.yaml")
        cfg.scale_factor = 4
        model = SRLatentDiffusion(cfg, device=device)
        print(f"Loading official OpenSR weights from HuggingFace ({cfg.ckpt_version})...")
        model.load_pretrained(cfg.ckpt_version)

        def _s2_only_encode(X_s2, X_s1):
            model._X_s2 = X_s2.clone()
            lr_size = X_s2.shape[-1]
            hr_size = lr_size * model.scale_factor
            X_s2_refl = X_s2.clamp(0, 1)
            X_s2_up = F.interpolate(X_s2_refl, size=(hr_size, hr_size),
                                    mode="bilinear", align_corners=False)
            cond_s2 = model.model.first_stage_model.encode(X_s2_up).sample()
            return cond_s2.to(model.device)
        def _s2_only_decode(latent, spe_cor=False):
            decoded = model.model.first_stage_model.decode(latent)
            return ((decoded + 1.0) / 2.0).clamp(0, 1) * 255.0

        model._tensor_encode = _s2_only_encode
        model._tensor_decode = _s2_only_decode
        hr_eval_size = HR_NATIVE  # 484 — 4× SR output size

    model.eval()

    s1_dummy = torch.zeros(1, 2, LR_PAD, LR_PAD, device=device)

    all_rows = []

    for ds_name in args.dataset:
        print(f"\nLoading {ds_name} dataset...")
        data = opensr_test.dataset.load(ds_name)

        l2a = data["L2A"][:, L2A_RGBNIR].astype(np.float32)  # (N, 4, 121, 121)
        hrm = data["HRharm"].astype(np.float32)               # (N, 4, 484, 484)
        N = l2a.shape[0]
        print(f"  {N} samples")

        rows = []

        for i in tqdm(range(N), desc=ds_name):
            s2 = torch.from_numpy(l2a[i]).unsqueeze(0)   # (1, 4, 121, 121) — [0,1] reflectance

            if use_custom:
                # User's model expects raw DN [0, 10000]; benchmark data is [0,1] reflectance
                s2_input = pad_right_bottom(s2 * 10000.0, LR_PAD)  # (1, 4, 128, 128)
            else:
                s2_input = pad_right_bottom(s2, LR_PAD)             # (1, 4, 128, 128)

            with torch.no_grad():
                sr: torch.Tensor = model.forward(
                    s2_input.to(device), s1_dummy,
                    sampling_steps=args.steps,
                    guidance_scale=args.guidance,
                    histogram_matching=False,
                )

            # Crop to native SR content (top-left: right-bottom padding was used)
            sr_crop = sr[:, :, :hr_eval_size, :hr_eval_size].cpu()  # (1, 4, hr_eval_size, hr_eval_size)

            hr_t = torch.from_numpy(hrm[i])  # (4, 484, 484)

            if use_custom:
                # Downsample 484×484 HR to 242×242 to match 2× SR scale
                hr_t = F.interpolate(hr_t.unsqueeze(0), size=(hr_eval_size, hr_eval_size),
                                     mode="bilinear", align_corners=False)[0]

            # Normalize to [0, 1] for opensr_test
            # l2a and hrm from opensr_test are already [0, 1] reflectance — use directly
            lr_norm = s2[0].clamp(0, 1)
            hr_norm = hr_t.clamp(0, 1)
            sr_norm = (sr_crop[0] / 255.0).clamp(0, 1)

            try:
                metrics = opensr_test.Metrics()
                result = metrics.compute(lr=lr_norm, sr=sr_norm, hr=hr_norm)
            except Exception as e:
                print(f"  [{ds_name}:{i}] opensr_test failed: {e}")
                result = {k: float("nan") for k in METRIC_KEYS[:7]}

            # PSNR + SSIM (RGB channels, uint8 scale)
            sr_u8 = (sr_norm[:3].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            hr_u8 = (hr_norm[:3].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            psnr = peak_signal_noise_ratio(hr_u8, sr_u8, data_range=255)
            ssim = structural_similarity(hr_u8, sr_u8, channel_axis=2, data_range=255)

            row = {"dataset": ds_name, "idx": i}
            for k in METRIC_KEYS[:7]:
                row[k] = float(result.get(k, float("nan")))
            row["psnr"] = float(psnr)
            row["ssim"] = float(ssim)
            rows.append(row)

        all_rows.extend(rows)

        print("\n" + "=" * 52)
        print(f"  {ds_name}  ({len(rows)} samples)")
        print("=" * 52)
        for k in METRIC_KEYS:
            vals = np.array([r[k] for r in rows], dtype=float)
            mean = np.nanmean(vals)
            std  = np.nanstd(vals)
            print(f"  {METRIC_LABELS[k]:<22}  {mean:8.4f} ± {std:.4f}")
        print("=" * 52)

    if len(args.dataset) > 1 and all_rows:
        print("\n" + "=" * 52)
        print(f"  Overall ({len(all_rows)} samples across {len(args.dataset)} datasets)")
        print("=" * 52)
        for k in METRIC_KEYS:
            vals = np.array([r[k] for r in all_rows], dtype=float)
            mean = np.nanmean(vals)
            std  = np.nanstd(vals)
            print(f"  {METRIC_LABELS[k]:<22}  {mean:8.4f} ± {std:.4f}")
        print("=" * 52)

    if args.out_csv and all_rows:
        out = pathlib.Path(args.out_csv)
        fieldnames = ["dataset", "idx"] + METRIC_KEYS
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nPer-sample results saved to {out}")


if __name__ == "__main__":
    main()
