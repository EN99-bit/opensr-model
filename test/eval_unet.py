"""UNet 'oracle' diagnostic: how well does the UNet recover the GT latent
given conditioning + a known amount of noise added to the target?

Analog of test/eval_vae.py for the diffusion UNet. The VAE ceiling answers
"how close to GT can the autoencoder get round-tripping a clean image?". This
script answers "how close to GT can the UNet get in a single step from a
noised target latent?" — an upper bound on the multi-step DDIM sampler.

Pipeline per tile, per timestep t:
    z_0 = scale * VAE.encode(GT_aerial).mode()              # scaled latent
    z_t = sqrt(a_bar_t) * z_0 + sqrt(1 - a_bar_t) * eps     # noise added at level t
    eps_hat = UNet(z_t, t, conditioning)                    # single-step prediction
    z_hat_0 = (z_t - sqrt(1 - a_bar_t) * eps_hat) / sqrt(a_bar_t)
    img_hat = VAE.decode(z_hat_0)
    score(img_hat, GT_aerial)                                # PSNR / SSIM / MAE

Sweep t across a few fractions of T (default 0.1, 0.25, 0.5, 0.75, 0.9) and
produce both a per-(tile, t) CSV and a 'PSNR vs t' line plot — the headline
report figure.

Works for both diffusion stages by passing the matching --config and --pad_size:
    - 1m UNet: --config opensr_model/configs/config_1m.yaml  --pad_size 1024
    - 5m UNet: --config opensr_model/configs/config_10m.yaml --pad_size 256

Usage:
    # 1m
    python test/eval_unet.py \
        --unet_ckpt checkpoints/1m/unet/last.ckpt \
        --npz_dir ~/npz/apr2025/1m-npz --pad_size 1024 --max_tiles 50 --save_plot

    # 5m
    python test/eval_unet.py \
        --unet_ckpt checkpoints/5m/unet-latents/last.ckpt \
        --config opensr_model/configs/config_10m.yaml \
        --npz_dir ~/npz/apr2025/5m-untouched --pad_size 256 --save_plot
"""

import argparse
import csv
import os
import pathlib
import re
import shlex
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opensr_model.srmodel import SRLatentDiffusion
from opensr_model.data import FusionDataset, LR_PAD_SIZE
from opensr_model.utils import normalize_aerial, normalize_s2, normalize_s1

# Reuse eval.py's UNet-weight loader (handles the 'ldm.' prefix strip; VAE bundled in ckpt)
from eval import load_trained_weights

# Reuse eval_cfg's plotting + metric set so this diagnostic matches the CFG sweep layout.
sys.path.insert(0, str(ROOT / "test"))
from eval_cfg import (
    save_metric_plots, save_tile_grid, lpips_score, OPENSR_METRICS, OM_HA_PANEL,
    PANEL_METRICS_FULL, PANEL_METRICS_PIXEL,
)

try:
    import opensr_test
except ImportError:
    opensr_test = None

# Per-tile metrics: pixel fidelity (incl. MAE), perceptual (LPIPS),
# and the opensr_test suite (om/ha/im, etc.).
UNET_METRICS = ["psnr", "ssim", "lpips", "mae", *OPENSR_METRICS]


def pad_to_size(x: torch.Tensor, size: int):
    """Symmetrically zero-pad (B,C,H,W) to (size, size). Returns (padded, pad_tuple)."""
    h, w = x.shape[-2:]
    ph, pw = size - h, size - w
    if ph < 0 or pw < 0:
        raise ValueError(f"pad size {size} smaller than tile {h}x{w}")
    pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)  # (l, r, t, b)
    return (F.pad(x, pad), pad) if (ph or pw) else (x, pad)


def crop_pad(x: torch.Tensor, pad) -> torch.Tensor:
    """Undo pad_to_size — return the native-sized center crop."""
    l, r, t, b = pad
    H, W = x.shape[-2:]
    return x[..., t:(H - b) if b else H, l:(W - r) if r else W]


def encode_target(model, x_norm):
    """Encode normalized aerial -> scaled latent z_0 used as the diffusion target."""
    posterior = model.model.first_stage_model.encode(x_norm)
    return model.model.scale_factor * posterior.mode()


def predict_x0(model, z_t, t_int, cond, a_bar_t):
    """Single-step x_0 prediction from a noised latent (eps-parameterization)."""
    t_batch = torch.full((z_t.shape[0],), t_int, device=z_t.device, dtype=torch.long)
    eps_hat = model.model.apply_model(z_t, t_batch, cond=cond)
    sqrt_a = a_bar_t.sqrt()
    sqrt_1ma = (1.0 - a_bar_t).sqrt()
    return (z_t - sqrt_1ma * eps_hat) / sqrt_a


AERIAL_KEYS = ("aerial_r", "aerial_g", "aerial_b", "aerial_nir")
AERIAL_5M_PAD = 256   # 5m aerial: 200×200 native → 256 (matches training)


def build_cond_5m_s1(model, aerial_5m, s1, pad_size):
    """Cascade stage-2 conditioning, matching train_unet_5-1_with_s1._build_conditioning:
    VAE(5m aerial upsampled to pad_size)[4ch] + S1[2ch] = 6ch.

    `aerial_5m` is (B,4,256,256) in [0,255]; `s1` is (B,2,128,128). Uses the VAE
    posterior mode() (deterministic) rather than sample(), since this is an oracle.
    """
    vae = model.model.first_stage_model
    vae_dtype = next(vae.parameters()).dtype
    latent = pad_size // model.vae_downscale
    aerial_up = F.interpolate(normalize_aerial(aerial_5m, stage="norm"),
                              size=(pad_size, pad_size), mode="bilinear", align_corners=False)
    with torch.no_grad():
        cond_5m = vae.encode(aerial_up.to(vae_dtype)).mode().float()
    cond_s1 = F.interpolate(normalize_s1(s1, stage="norm"),
                            size=(latent, latent), mode="bilinear", align_corners=False)
    return torch.cat([cond_5m, cond_s1], dim=1)


def to_rgb_u8(img_norm):
    """[-1,1] (1,4,H,W) tensor -> RGB uint8 (H,W,3) for skimage metrics."""
    arr = normalize_aerial(img_norm, stage="denorm")[0, :3].cpu().numpy().transpose(1, 2, 0)
    return arr.clip(0, 255).astype(np.uint8)


def resolve_run_dir(args):
    """Per-UNet output subdir: <out_dir>/<raw-stage>_<ckpt-stem>/.

    Returns (run_dir, display_stage). The folder uses the literal path
    component (so "5to1m_with_s2_last" and "1m_last" stay distinct), but the
    display_stage normalizes cascade names ("5to1m..." → "1m") for the title.
    """
    ckpt_path = pathlib.Path(args.unet_ckpt)
    # Match either a plain resolution ("1m", "5m") or a cascade variant ("5to1m", "5to1m_with_s2").
    raw_stage = next(
        (p for p in ckpt_path.parts
         if re.fullmatch(r"\d+m", p) or re.match(r"^\d+to\d+m", p)),
        None,
    )
    label = f"{raw_stage}_{ckpt_path.stem}" if raw_stage else ckpt_path.stem
    run_dir = pathlib.Path(args.out_dir) / label
    run_dir.mkdir(parents=True, exist_ok=True)

    # Cascade naming ("5to1m...") describes input→output stages; the *output* (target
    # resolution after the to) is what the model actually produces, so title shows that.
    display_stage = raw_stage
    if raw_stage:
        m = re.match(r"^\d+to(\d+)m", raw_stage)
        if m:
            display_stage = f"{m.group(1)}m"
    return run_dir, display_stage


def run_eval_loop(args, t_fracs, shard_idx=0, shard_count=1):
    """Per-tile evaluation. With shard striping for multi-GPU fan-out.
    Returns (rows, T)."""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"[shard {shard_idx}/{shard_count}]" if shard_count > 1 else ""
    print(f"{tag} Device: {device}")

    cfg = OmegaConf.load(args.config)
    model = SRLatentDiffusion(cfg, device=device)
    print(f"{tag} Loading UNet (+ bundled VAE): {args.unet_ckpt}")
    load_trained_weights(model, args.unet_ckpt)
    model = model.to(device).eval()
    T = int(model.model.num_timesteps)
    print(f"{tag} Model: T={T}, parameterization={model.model.parameterization}, "
          f"scale_factor={float(model.model.scale_factor):.5f}")
    assert model.model.parameterization == "eps", \
        f"Only eps-parameterization is implemented (got {model.model.parameterization})"

    ds = FusionDataset(args.npz_dir, require_aerial=True, pad=False)
    if len(ds) == 0:
        print("No valid tiles found.")
        return [], T
    n_total = len(ds) if args.max_tiles is None else min(args.max_tiles, len(ds))
    tile_indices = list(range(shard_idx, n_total, shard_count))
    print(f"{tag} Evaluating {len(tile_indices)} of {n_total} tiles at t-fractions {t_fracs}")

    # Cascade stage-2 (5to1m) models condition on the 5m aerial, not S2. When a 5m
    # tile dir is given, build conditioning from the paired 5m aerial + S1 (matching
    # training); otherwise use the model's default S2+S1 encoding.
    cond_5m_map = None
    if getattr(args, "cond_5m_dir", None):
        cond_5m_map = {p.stem: p for p in sorted(pathlib.Path(args.cond_5m_dir).glob("*.npz"))}
        if not cond_5m_map:
            print(f"No .npz tiles in --cond_5m_dir {args.cond_5m_dir}")
            return [], int(OmegaConf.load(args.config).denoiser_settings.timesteps)
        print(f"{tag} Cascade conditioning: VAE(5m aerial)+S1 from {args.cond_5m_dir} "
              f"({len(cond_5m_map)} tiles)")

    save_images = getattr(args, "save_images", False)
    if save_images:
        images_dir, _ = resolve_run_dir(args)
        images_dir = images_dir / "images"
        images_dir.mkdir(exist_ok=True)

    # Only shard 0 shows the progress bar to avoid 4 interleaved tqdm streams.
    show_progress = (shard_idx == 0)
    rows = []
    for i in tqdm(tile_indices, desc=f"UNet oracle {tag}".strip(), disable=not show_progress):
        sample = ds[i]
        name = pathlib.Path(sample["path"]).stem

        aerial = sample["aerial"].unsqueeze(0).to(device)
        s2 = sample["s2"].unsqueeze(0).to(device)
        s1 = sample["s1"].unsqueeze(0).to(device)

        with torch.no_grad():
            aerial_p, aerial_pad = pad_to_size(aerial, args.pad_size)
            x_norm = normalize_aerial(aerial_p, stage="norm")
            z_0 = encode_target(model, x_norm)
            s2_p, _ = pad_to_size(s2, LR_PAD_SIZE)
            s1_p, _ = pad_to_size(s1, LR_PAD_SIZE)
            if cond_5m_map is not None:
                # Build cascade conditioning from the paired 5m aerial + S1.
                p5 = cond_5m_map.get(name)
                if p5 is None:
                    if show_progress:
                        print(f"  [skip] no 5m tile for {name}")
                    continue
                with np.load(p5, allow_pickle=True) as d5:
                    aerial_5m = torch.from_numpy(
                        np.stack([d5[k].astype(np.float32) for k in AERIAL_KEYS])
                    ).unsqueeze(0).to(device)
                aerial_5m, _ = pad_to_size(aerial_5m, AERIAL_5M_PAD)
                cond = build_cond_5m_s1(model, aerial_5m, s1_p, args.pad_size)
            else:
                cond = model._tensor_encode(s2_p, s1_p)
            assert z_0.shape[-2:] == cond.shape[-2:], (
                f"latent/cond spatial mismatch {z_0.shape} vs {cond.shape}"
            )
            # Per-tile noise seed (shard-independent: same tile always gets same eps).
            gi = torch.Generator(device="cpu").manual_seed(args.seed + i)
            eps = torch.randn(z_0.shape, generator=gi).to(device)

            # Reference tensors for perceptual (LPIPS) + opensr_test scoring.
            # HR is constant per tile; the S2 LR is the opensr_test reference
            # (satalign breaks for scale > 4, so upsample LR to keep scale ≤ 4).
            hr_norm = (aerial[0] / 255.0).clamp(0, 1)
            run_opensr = (opensr_test is not None) and (not getattr(args, "no_opensr_test", False))
            if run_opensr:
                lr_eval = (normalize_s2(s2[0].float(), stage="norm") + 1.0) / 2.0
                if hr_norm.shape[-1] / lr_eval.shape[-1] > 4:
                    tgt = hr_norm.shape[-1] // 4
                    lr_eval = F.interpolate(lr_eval.unsqueeze(0), size=(tgt, tgt),
                                            mode="bilinear", align_corners=False)[0]

            tile_thumbs = {} if save_images else None
            for frac in t_fracs:
                t_int = max(0, min(T - 1, int(round(frac * (T - 1)))))
                a_bar_t = model.model.alphas_cumprod[t_int].to(device)
                z_t = a_bar_t.sqrt() * z_0 + (1.0 - a_bar_t).sqrt() * eps
                z_hat_0 = predict_x0(model, z_t, t_int, cond, a_bar_t)

                img_hat = model.model.decode_first_stage(z_hat_0)
                img_hat_n = crop_pad(img_hat, aerial_pad)
                x_native = normalize_aerial(aerial, stage="norm")

                mae = torch.mean(torch.abs(x_native - img_hat_n)).item()
                hr_u8 = to_rgb_u8(x_native)
                sr_u8 = to_rgb_u8(img_hat_n)
                if tile_thumbs is not None:
                    tile_thumbs[frac] = sr_u8
                psnr = peak_signal_noise_ratio(hr_u8, sr_u8, data_range=255)
                ssim = structural_similarity(hr_u8, sr_u8, channel_axis=2, data_range=255)

                sr_norm = (normalize_aerial(img_hat_n, stage="denorm")[0] / 255.0).clamp(0, 1)
                row = {"tile": name, "t_frac": frac, "t": t_int,
                       "psnr": float(psnr), "ssim": float(ssim), "mae": float(mae),
                       "lpips": lpips_score(sr_norm, hr_norm, device)}
                if run_opensr:
                    try:
                        r = opensr_test.Metrics().compute(lr=lr_eval, sr=sr_norm, hr=hr_norm)
                        for k in OPENSR_METRICS:
                            row[k] = float(r.get(k, float("nan")))
                    except Exception as e:
                        if show_progress:
                            print(f"  opensr_test failed (tile={name}, t={t_int}): {e}")
                        for k in OPENSR_METRICS:
                            row[k] = float("nan")
                else:
                    for k in OPENSR_METRICS:
                        row[k] = float("nan")
                rows.append(row)

            if tile_thumbs is not None:
                save_tile_grid(name, hr_u8, tile_thumbs, "UNet-orakel", images_dir / f"{name}.png",
                               label_fmt="t/T={:g}")

    return rows, T


def aggregate_per_t(rows, t_fracs):
    """{t_frac: {metric: (mean, std)}} across tiles — same shape save_one_curve expects."""
    per_t = {}
    for frac in t_fracs:
        sel = [r for r in rows if r["t_frac"] == frac]
        per_t[frac] = {}
        for k in UNET_METRICS:
            v = np.array([r[k] for r in sel], dtype=np.float64)
            v = v[~np.isnan(v)]
            per_t[frac][k] = (float(v.mean()), float(v.std())) if len(v) else (float("nan"), float("nan"))
    return per_t


def write_final_outputs(args, rows, run_dir, t_fracs, T, stage, cmd_line):
    """Write metrics.csv (cmd at top, per-(tile,t) rows, then per-t mean/std rows),
    print the summary, and optionally save the PSNR-vs-t plot."""
    per_t = aggregate_per_t(rows, t_fracs)
    n_tiles = len({r["tile"] for r in rows})
    no_opensr = getattr(args, "no_opensr_test", False)

    out_csv = run_dir / "metrics.csv"
    fields = ["tile", "t_frac", "t", *UNET_METRICS]
    with open(out_csv, "w", newline="") as f:
        f.write(f"# {cmd_line}\n")
        f.write(f"# effective: seed={args.seed} opensr_test={'off' if no_opensr else 'on'}\n")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        for frac in t_fracs:
            t_int = max(0, min(T - 1, int(round(frac * (T - 1)))))
            writer.writerow({"tile": f"mean_t={frac}", "t_frac": frac, "t": t_int,
                             **{k: per_t[frac][k][0] for k in UNET_METRICS}})
            writer.writerow({"tile": f"std_t={frac}", "t_frac": frac, "t": t_int,
                             **{k: per_t[frac][k][1] for k in UNET_METRICS}})

    table_keys = ["psnr", "ssim", "lpips", "mae"] + ([] if no_opensr else ["im_metric", "ha_metric", "om_metric"])
    print(f"\n  UNet oracle reconstruction over {n_tiles} tiles  (eps-pred, seed={args.seed})")
    print(f"  {'t/T':>5}  {'t':>4}  " + "  ".join(f"{k:>10}" for k in table_keys))
    for frac in t_fracs:
        t_int = max(0, min(T - 1, int(round(frac * (T - 1)))))
        vals = "  ".join(f"{per_t[frac][k][0]:10.4f}" for k in table_keys)
        print(f"  {frac:>5.2f}  {t_int:>4d}  {vals}")
    print(f"\nPer-tile-per-t results saved to {out_csv}")

    if args.save_plot:
        # One plot file per metric in oracle_plots/. No baseline — noise level
        # has no "no-guidance" reference point.
        metrics_to_plot = (["psnr", "ssim", "lpips", "mae"]
                           if no_opensr else UNET_METRICS)
        plots_dir = run_dir / "oracle_plots"
        save_metric_plots(per_t, t_fracs, plots_dir, "Støjniveau (t / T)",
                          metrics_to_plot, baseline_x=None)
        print(f"Per-metric plots saved to {plots_dir}/ ({len(metrics_to_plot)} metrics)")


def write_shard_csv(rows, run_dir, shard_idx):
    """Per-shard partial CSV (no command line, no aggregates). Merged by the parent."""
    out = run_dir / f"metrics.shard{shard_idx}.csv"
    fields = ["tile", "t_frac", "t", *UNET_METRICS]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[shard {shard_idx}] wrote {out} ({len(rows)} rows)")


def read_shard_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {"tile": r["tile"], "t_frac": float(r["t_frac"]), "t": int(r["t"])}
            for k in UNET_METRICS:
                row[k] = float(r[k])
            rows.append(row)
    return rows


def run_multi_gpu(args, t_fracs, n_devices):
    """Fan out across N GPUs via N child processes pinned to one GPU each, then merge."""
    run_dir, stage = resolve_run_dir(args)
    cmd_line = "python " + shlex.join(sys.argv)
    script = str(pathlib.Path(__file__).resolve())

    print(f"Multi-GPU mode: launching {n_devices} shard processes")
    procs = []
    for i in range(n_devices):
        cmd = [sys.executable, script,
               "--unet_ckpt", args.unet_ckpt,
               "--npz_dir", args.npz_dir,
               "--config", args.config,
               "--pad_size", str(args.pad_size),
               "--t_fracs", args.t_fracs,
               "--out_dir", args.out_dir,
               "--seed", str(args.seed),
               "--shard", f"{i}/{n_devices}",
               "--devices", "1",
               "--device", "cuda:0"]  # masked by CUDA_VISIBLE_DEVICES below
        if args.max_tiles is not None:
            cmd += ["--max_tiles", str(args.max_tiles)]
        if args.no_opensr_test:
            cmd += ["--no_opensr_test"]
        if args.save_images:
            cmd += ["--save_images"]
        if args.cond_5m_dir:
            cmd += ["--cond_5m_dir", args.cond_5m_dir]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # Inherit stdout so shard 0's tqdm shows live; other shards print silently.
        procs.append((i, subprocess.Popen(cmd, env=env)))

    failures = [i for i, p in procs if p.wait() != 0]
    if failures:
        print(f"ERROR: shard(s) {failures} failed.")
        sys.exit(1)

    # Merge
    all_rows = []
    for i in range(n_devices):
        shard_csv = run_dir / f"metrics.shard{i}.csv"
        all_rows.extend(read_shard_csv(shard_csv))
        shard_csv.unlink()
    # T from config (no model loaded in parent)
    T = int(OmegaConf.load(args.config).denoiser_settings.timesteps)
    write_final_outputs(args, all_rows, run_dir, t_fracs, T, stage, cmd_line)


def main():
    parser = argparse.ArgumentParser(description="UNet x0-oracle quality vs noise-level diagnostic")
    parser.add_argument("--unet_ckpt", required=True, help="Path to UNet .ckpt (Lightning state_dict)")
    parser.add_argument("--npz_dir", required=True, help="Directory of .npz tiles with aerial bands")
    parser.add_argument("--config", default=str(ROOT / "opensr_model" / "configs" / "config_1m.yaml"),
                        help="Config YAML matching the UNet "
                             "(1m: config_1m.yaml; 5m: config_10m.yaml)")
    parser.add_argument("--pad_size", type=int, required=True,
                        help="Zero-pad aerial tiles to this size (1024 for 1m, 256 for 5m).")
    parser.add_argument("--cond_5m_dir", type=str, default=None,
                        help="For cascade stage-2 (5to1m) models: directory of 5m NPZ tiles. "
                             "Conditioning is then VAE(5m aerial)+S1 (matching training) instead "
                             "of the default S2+S1. Tiles are paired with --npz_dir by filename stem.")
    parser.add_argument("--t_fracs", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                        help="Comma-separated fractions of T to test (each in (0,1))")
    parser.add_argument("--out_dir", default=str(ROOT / "test" / "results" / "unet-oracle"))
    parser.add_argument("--max_tiles", type=int, default=None)
    parser.add_argument("--save_plot", action="store_true",
                        help="Save per-metric plots into oracle_plots/")
    parser.add_argument("--save_images", action="store_true",
                        help="Save a per-tile grid (Original + reconstruction@each t/T) "
                             "under <run_dir>/images/.")
    parser.add_argument("--no_opensr_test", action="store_true",
                        help="Skip the opensr_test metrics (reflectance/spectral/om/ha/im); "
                             "plot then shows only PSNR + SSIM.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--devices", type=int,
                        default=(torch.cuda.device_count() if torch.cuda.is_available() else 1),
                        help="Number of GPUs to use via subprocess sharding "
                             "(default: all visible CUDA devices; set 1 for single-GPU).")
    parser.add_argument("--shard", type=str, default=None,
                        help="i/N stripe selector (internal; set by the multi-GPU launcher).")
    args = parser.parse_args()

    t_fracs = sorted(float(s) for s in args.t_fracs.split(","))
    assert all(0 < f < 1 for f in t_fracs), "--t_fracs values must be in (0, 1)"

    # Dispatch ------------------------------------------------------------------
    # 1) Shard mode (called by the multi-GPU launcher): process stripe, write partial CSV.
    if args.shard is not None:
        shard_idx, shard_count = (int(x) for x in args.shard.split("/"))
        rows, _ = run_eval_loop(args, t_fracs, shard_idx, shard_count)
        run_dir, _ = resolve_run_dir(args)
        write_shard_csv(rows, run_dir, shard_idx)
        return

    # 2) Multi-GPU fan-out: launch N child processes, then merge their CSVs.
    if args.devices > 1:
        run_multi_gpu(args, t_fracs, args.devices)
        return

    # 3) Single-GPU (or CPU) path.
    rows, T = run_eval_loop(args, t_fracs)
    run_dir, stage = resolve_run_dir(args)
    cmd_line = "python " + shlex.join(sys.argv)
    write_final_outputs(args, rows, run_dir, t_fracs, T, stage, cmd_line)


if __name__ == "__main__":
    main()
