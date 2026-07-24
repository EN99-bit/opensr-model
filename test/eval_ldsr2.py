"""Evaluate the official LDSR-S2 baseline and save ALL metrics to one CSV.

Loads the published OpenSR LDSR-S2 weights (S2-only, 4x: 10 m -> 2.5 m) and runs
them on a directory of NPZ test tiles. For every tile it computes the full
opensr-test suite together with PSNR, SSIM and LPIPS, so the baseline can be
compared in the same format as the trained 5 m and cascade models.

Resolution note. LDSR-S2 produces a 2.5 m image, but the ground truth in these
tiles is 5 m. To compare against real ground truth, the 2.5 m output is averaged
down to 5 m so it covers the same 1 km footprint as the GT. The comparison across
the three systems is therefore at different native scales (5 m model -> 5 m,
LDSR-S2 -> 2.5 m downsampled to 5 m, cascade -> 1 m), which is context rather than
a head-to-head test, but the shared format and metric set still make it useful.

Requires opensr-test and lpips.

Usage:
    python test/eval_ldsr2.py
    python test/eval_ldsr2.py --npz_dir ~/npz/apr2025/5m-untouched --device cpu
    python test/eval_ldsr2.py --save_images
"""

import argparse
import csv
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
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

from opensr_model.srmodel import SRLatentDiffusion
from opensr_model.data import FusionDataset, LR_PAD_SIZE, HR_PAD_SIZE
from opensr_model.utils import normalize_s2

DEFAULT_INPUT_DIR = pathlib.Path("~/npz/apr2025/5m-untouched").expanduser()

ORIG_LR = 100   # S2 native pixels per tile (10 m)
ORIG_HR = 200   # aerial GT native pixels per tile (5 m)
SCALE   = 4     # official model is 4x (10 m -> 2.5 m)

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
    """LPIPS (VGG) between SR and HR, RGB only. Inputs (C,H,W) in [0,1]; lower is better."""
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
    if t.dim() == 4:
        t = t[0]
    img = np.clip(t[:3].cpu().numpy(), 0, 255).astype(np.uint8)
    return np.transpose(img, (1, 2, 0))


def hist_match(sr, hr):
    """Per-channel histogram-match SR to the GT colour distribution. Both (C,H,W) in [0,1].

    Removes the global S2-vs-aerial colour offset so the metrics measure detail and
    structure rather than the colour translation the trained models learned.
    """
    sr_np = sr.cpu().numpy().transpose(1, 2, 0)
    hr_np = hr.cpu().numpy().transpose(1, 2, 0)
    matched = match_histograms(sr_np, hr_np, channel_axis=-1)
    return torch.from_numpy(np.ascontiguousarray(matched.transpose(2, 0, 1))).float()


def build_ldsr2(device):
    """Load the official LDSR-S2 weights with S2-only 4x conditioning."""
    cfg = OmegaConf.load(ROOT / "opensr_model" / "configs" / "config_opensr.yaml")
    model = SRLatentDiffusion(cfg, device=device)
    print(f"Loading official OpenSR weights from HuggingFace ({cfg.ckpt_version})...")
    model.load_pretrained(cfg.ckpt_version)
    model.scale_factor = SCALE  # config defaults to 2; official model is 4x

    def _encode(X_s2, X_s1=None):
        model._X_s2 = X_s2.clone()
        hr_size = X_s2.shape[-1] * model.scale_factor
        X_s2_refl = (X_s2 / 10000.0).clamp(0, 1)  # official model expects [0,1] reflectance
        X_s2_up = F.interpolate(X_s2_refl, size=(hr_size, hr_size),
                                mode="bilinear", align_corners=False)
        return model.model.first_stage_model.encode(X_s2_up).sample().to(model.device)

    def _decode(latent, spe_cor=False):
        # Official model trained with scale_factor 1.0; decode directly to avoid the
        # erroneous 1/0.18215 scaling in srmodel.py that would saturate the output.
        decoded = model.model.first_stage_model.decode(latent)
        return ((decoded + 1.0) / 2.0).clamp(0, 1) * 255.0

    model._tensor_encode = _encode
    model._tensor_decode = _decode
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate the LDSR-S2 baseline with all metrics")
    parser.add_argument("--npz_dir", type=str, default=str(DEFAULT_INPUT_DIR),
                        help="Directory containing .npz test tiles")
    parser.add_argument("--steps", type=int, default=100, help="DDIM sampling steps")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None,
                        help="Device (default: auto-detect)")
    parser.add_argument("--out_csv", type=str, default=None,
                        help="CSV path (default: test/results/ldsr2_baseline/metrics.csv)")
    parser.add_argument("--save_images", action="store_true",
                        help="Also save S2 | SR(5m) | GT comparison PNGs")
    parser.add_argument("--histogram_match", action="store_true",
                        help="Histogram-match the SR to the GT colours before scoring, so the "
                             "metrics measure detail rather than the S2-vs-aerial colour gap")
    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    default_dir = "ldsr2_baseline_histmatch" if args.histogram_match else "ldsr2_baseline"
    out_csv = pathlib.Path(args.out_csv) if args.out_csv else \
        ROOT / "test" / "results" / default_dir / "metrics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    model = build_ldsr2(device)

    ds = FusionDataset(args.npz_dir, require_aerial=True, pad=True)
    if len(ds) == 0:
        print("No valid tiles found.")
        return
    print(f"Evaluating {len(ds)} tiles (LDSR-S2, 4x, then downsampled 2.5 m -> 5 m)...")

    lr_pad = (LR_PAD_SIZE - ORIG_LR) // 2          # 14
    hr_pad = (HR_PAD_SIZE - ORIG_HR) // 2          # GT 5 m offset in padded aerial
    sr_off = lr_pad * SCALE                         # native content offset in 4x output (56)
    sr_native = ORIG_LR * SCALE                     # native content size in 4x output (400)

    metrics_engine = opensr_test.Metrics()
    rows = []

    for i in tqdm(range(len(ds)), desc="LDSR-S2"):
        sample = ds[i]
        name = pathlib.Path(sample["path"]).stem
        s2     = sample["s2"].unsqueeze(0)
        s1     = sample["s1"].unsqueeze(0)
        aerial = sample["aerial"].unsqueeze(0)

        with torch.no_grad():
            sr = model.forward(s2.to(device), s1.to(device),
                               sampling_steps=args.steps, guidance_scale=1.0,
                               histogram_matching=False)

        # Full 2.5 m content (400x400), then average down to 5 m (200x200) to match GT
        sr_25 = sr[:, :, sr_off:sr_off + sr_native, sr_off:sr_off + sr_native].cpu()
        sr_5m = F.interpolate(sr_25.float(), size=(ORIG_HR, ORIG_HR), mode="area")

        s2_crop     = s2[:, :, lr_pad:lr_pad + ORIG_LR, lr_pad:lr_pad + ORIG_LR]
        aerial_crop = aerial[:, :, hr_pad:hr_pad + ORIG_HR, hr_pad:hr_pad + ORIG_HR]

        lr_norm = (normalize_s2(s2_crop[0], stage="norm") + 1.0) / 2.0  # (4,100,100) in [0,1]
        sr_norm = (sr_5m[0] / 255.0).clamp(0, 1)                        # (4,200,200)
        hr_norm = (aerial_crop[0] / 255.0).clamp(0, 1)                  # (4,200,200)

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
            s2_up = F.interpolate(s2_crop.float(), size=(ORIG_HR, ORIG_HR),
                                  mode="bilinear", align_corners=False)
            s2_np = s2_up[0, :3].numpy()
            lo, hi = np.percentile(s2_np, 2), np.percentile(s2_np, 98)
            s2_img = to_rgb(((s2_up - lo) / (hi - lo + 1e-6)).clamp(0, 1) * 255)
            if args.histogram_match:
                # Already matched to GT colours -> show raw, no stretch needed
                sr_img = to_rgb(sr_norm[:3] * 255)
            else:
                # LDSR-S2 outputs S2 reflectance (narrow range) -> percentile-stretch like S2,
                # otherwise the panel looks flat gray even though the SR is valid.
                sr_np = sr_5m[0, :3].numpy()
                slo, shi = np.percentile(sr_np, 2), np.percentile(sr_np, 98)
                sr_img = to_rgb(((sr_5m - slo) / (shi - slo + 1e-6)).clamp(0, 1) * 255)
            gt_img = to_rgb(aerial_crop)
            Image.fromarray(np.concatenate([s2_img, sr_img, gt_img], axis=1)).save(
                img_dir / f"{name}.png")

    # Aggregate
    print("\n" + "=" * 52)
    print(f"  LDSR-S2 baseline over {len(rows)} tiles")
    print("=" * 52)
    agg = {}
    for k in METRIC_KEYS:
        vals = np.array([r[k] for r in rows], dtype=float)
        agg[k] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
        print(f"  {METRIC_LABELS[k]:<18}  {agg[k][0]:8.4f} ± {agg[k][1]:.4f}")
    print("=" * 52)

    with open(out_csv, "w", newline="") as f:
        cmd = "python " + " ".join(sys.argv)
        f.write(f"# {cmd}\n")
        writer = csv.DictWriter(f, fieldnames=["tile"] + METRIC_KEYS)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"tile": "mean", **{k: agg[k][0] for k in METRIC_KEYS}})
        writer.writerow({"tile": "std",  **{k: agg[k][1] for k in METRIC_KEYS}})
    print(f"\nSaved per-tile metrics + mean/std to {out_csv}")


if __name__ == "__main__":
    main()
