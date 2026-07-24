"""Bicubic baseline: the trivial 'do nothing clever' floor for the SR task.

Instead of running a model, we simply bicubic-upsample the 10 m Sentinel-2 input to
the 5 m aerial grid and score it against the ground truth with the SAME metrics and
CSV format as the model evaluations (eval_ldsr2.py / eval_vae.py). Any real model has
to beat this baseline to justify itself.

What to expect, and why:
  - PSNR / SSIM / LPIPS are computed between the upsampled S2 and the aerial GT. Bicubic
    upsampling stays in the Sentinel-2 reflectance domain, so these are dominated by the
    S2-vs-aerial domain gap (same caveat as the LDSR-S2 comparison) plus the blur of a
    plain interpolation. This is the floor the 5 m model improves on.
  - On the opensr-test correctness metrics the bicubic image is, by construction, the
    reference the framework itself upsamples from the LR. It adds no new detail, so
    improvement and hallucination go to ~0 and omission to ~1. That is the correct,
    interpretable reading of a trivial baseline: it omits all the detail and invents none.

Requires opensr-test and lpips.

Usage:
    python test/eval_bicubic.py --npz_dir ~/npz/apr2025/5m-untouched
    python test/eval_bicubic.py --npz_dir ~/npz/apr2025/5m-untouched --device cpu --save_images
"""

import argparse
import csv
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from skimage.exposure import match_histograms
from tqdm import tqdm

try:
    import opensr_test
except ImportError:
    print("opensr-test not installed. Run: pip install opensr-test")
    sys.exit(1)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opensr_model.data import FusionDataset
from opensr_model.utils import normalize_s2

DEFAULT_INPUT_DIR = pathlib.Path("~/npz/apr2025/5m-untouched").expanduser()

ORIG_LR = 100   # S2 native pixels per tile (10 m)
ORIG_HR = 200   # aerial GT native pixels per tile (5 m)

OPENSR_KEYS = ["reflectance", "spectral", "spatial", "synthesis",
               "ha_metric", "om_metric", "im_metric"]
METRIC_KEYS = ["psnr", "ssim", "lpips"] + OPENSR_KEYS

METRIC_LABELS = {
    "psnr": "PSNR ↑", "ssim": "SSIM ↑", "lpips": "LPIPS ↓",
    "reflectance": "Reflectance ↓", "spectral": "Spectral ↓",
    "spatial": "Spatial ↓", "synthesis": "Synthesis ↑",
    "ha_metric": "Hallucination ↓", "om_metric": "Omission ↓",
    "im_metric": "Improvement ↑",
}

_LPIPS_MODEL = None  # cached singleton; False if unavailable


def lpips_score(sr_norm, hr_norm, device):
    """LPIPS (VGG), RGB only. Inputs (C,H,W) in [0,1]; lower is better."""
    global _LPIPS_MODEL
    if _LPIPS_MODEL is False:
        return float("nan")
    if _LPIPS_MODEL is None:
        try:
            import lpips
            _LPIPS_MODEL = lpips.LPIPS(net="vgg").to(device).eval()
            for p in _LPIPS_MODEL.parameters():
                p.requires_grad = False
        except Exception as e:
            print(f"  LPIPS unavailable ({e}); reporting NaN for lpips.")
            _LPIPS_MODEL = False
            return float("nan")
    sr = sr_norm[:3].unsqueeze(0).to(device) * 2.0 - 1.0
    hr = hr_norm[:3].unsqueeze(0).to(device) * 2.0 - 1.0
    with torch.no_grad():
        return float(_LPIPS_MODEL(sr, hr).item())


def to_rgb(t):
    img = np.clip((t[:3].cpu().numpy() * 255), 0, 255).astype(np.uint8)
    return np.transpose(img, (1, 2, 0))


def hist_match(sr, hr):
    """Per-channel histogram-match SR to the GT colour distribution. Both (C,H,W) in [0,1]."""
    sr_np = sr.cpu().numpy().transpose(1, 2, 0)
    hr_np = hr.cpu().numpy().transpose(1, 2, 0)
    matched = match_histograms(sr_np, hr_np, channel_axis=-1)
    return torch.from_numpy(np.ascontiguousarray(matched.transpose(2, 0, 1))).float()


def main():
    parser = argparse.ArgumentParser(description="Bicubic baseline eval with all metrics")
    parser.add_argument("--npz_dir", type=str, default=str(DEFAULT_INPUT_DIR),
                        help="Directory containing .npz test tiles")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None,
                        help="Device for LPIPS (default: auto-detect)")
    parser.add_argument("--out_csv", type=str, default=None,
                        help="CSV path (default: test/results/bicubic_baseline/metrics.csv)")
    parser.add_argument("--save_images", action="store_true",
                        help="Also save bicubic | GT comparison PNGs")
    parser.add_argument("--histogram_match", action="store_true",
                        help="Histogram-match the bicubic output to the GT colours before scoring, "
                             "so the metrics measure detail rather than the S2-vs-aerial colour gap")
    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    default_dir = "bicubic_baseline_histmatch" if args.histogram_match else "bicubic_baseline"
    out_csv = pathlib.Path(args.out_csv) if args.out_csv else \
        ROOT / "test" / "results" / default_dir / "metrics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # pad=False -> native sizes: S2 100x100, aerial 200x200 (no padding to crop)
    ds = FusionDataset(args.npz_dir, require_aerial=True, pad=False)
    if len(ds) == 0:
        print("No valid tiles found.")
        return
    print(f"Evaluating {len(ds)} tiles (bicubic upsample 10 m -> 5 m)...")

    metrics_engine = opensr_test.Metrics()
    rows = []

    for i in tqdm(range(len(ds)), desc="Bicubic"):
        sample = ds[i]
        name = pathlib.Path(sample["path"]).stem
        s2     = sample["s2"]                 # (4, 100, 100) raw DN
        aerial = sample["aerial"]             # (4, 200, 200) [0, 255]

        # S2 -> [0, 1], then bicubic-upsample to the 5 m grid (clamp: bicubic can overshoot)
        lr_norm = (normalize_s2(s2, stage="norm") + 1.0) / 2.0          # (4,100,100) in [0,1]
        sr_norm = F.interpolate(lr_norm.unsqueeze(0), size=(ORIG_HR, ORIG_HR),
                                mode="bicubic", align_corners=False)[0].clamp(0, 1)  # (4,200,200)
        hr_norm = (aerial / 255.0).clamp(0, 1)                          # (4,200,200)

        if args.histogram_match:
            sr_norm = hist_match(sr_norm, hr_norm)

        try:
            res = metrics_engine.compute(lr=lr_norm, sr=sr_norm, hr=hr_norm)
        except Exception as e:
            print(f"  [{name}] opensr_test failed: {e}")
            res = {k: float("nan") for k in OPENSR_KEYS}

        sr_u8 = (sr_norm[:3].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        hr_u8 = (hr_norm[:3].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)

        row = {"tile": name}
        row["psnr"]  = float(peak_signal_noise_ratio(hr_u8, sr_u8, data_range=255))
        row["ssim"]  = float(structural_similarity(hr_u8, sr_u8, channel_axis=2, data_range=255))
        row["lpips"] = lpips_score(sr_norm, hr_norm, device)
        for k in OPENSR_KEYS:
            row[k] = float(res.get(k, float("nan")))
        rows.append(row)

        if args.save_images:
            img_dir = out_csv.parent / "images"
            img_dir.mkdir(exist_ok=True)
            Image.fromarray(np.concatenate(
                [to_rgb(sr_norm), to_rgb(hr_norm)], axis=1)).save(img_dir / f"{name}.png")

    # Aggregate
    print("\n" + "=" * 52)
    print(f"  Bicubic baseline over {len(rows)} tiles")
    print("=" * 52)
    agg = {}
    for k in METRIC_KEYS:
        vals = np.array([r[k] for r in rows], dtype=float)
        agg[k] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
        print(f"  {METRIC_LABELS[k]:<18}  {agg[k][0]:8.4f} ± {agg[k][1]:.4f}")
    print("=" * 52)

    with open(out_csv, "w", newline="") as f:
        f.write("# python " + " ".join(sys.argv) + "\n")
        writer = csv.DictWriter(f, fieldnames=["tile"] + METRIC_KEYS)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"tile": "mean", **{k: agg[k][0] for k in METRIC_KEYS}})
        writer.writerow({"tile": "std",  **{k: agg[k][1] for k in METRIC_KEYS}})
    print(f"\nSaved per-tile metrics + mean/std to {out_csv}")


if __name__ == "__main__":
    main()
