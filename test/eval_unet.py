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
from opensr_model.utils import normalize_aerial

# Reuse eval.py's UNet-weight loader (handles the 'ldm.' prefix strip; VAE bundled in ckpt)
from eval import load_trained_weights


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


def to_rgb_u8(img_norm):
    """[-1,1] (1,4,H,W) tensor -> RGB uint8 (H,W,3) for skimage metrics."""
    arr = normalize_aerial(img_norm, stage="denorm")[0, :3].cpu().numpy().transpose(1, 2, 0)
    return arr.clip(0, 255).astype(np.uint8)


def save_curve(per_t_stats, out_path, title):
    """Line plot: mean PSNR vs t-fraction with ±std band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fracs = sorted(per_t_stats.keys())
    psnr_mean = [per_t_stats[f]["psnr_mean"] for f in fracs]
    psnr_std = [per_t_stats[f]["psnr_std"] for f in fracs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(fracs, psnr_mean, marker="o", lw=2, label="Gennemsnit af testbilleder")
    ax.legend(loc="upper right", frameon=True)
    ax.set_xlabel("Støjniveau (t / T)")
    ax.set_ylabel("PSNR ↑")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    # Axis-break marks on the y-axis (╱╱) — signals that the y-axis does not start at 0.
    d = 0.015
    kw = dict(transform=ax.transAxes, color="k", lw=1, clip_on=False)
    ax.plot([-d, +d], [0.03 - d, 0.03 + d], **kw)
    ax.plot([-d, +d], [0.06 - d, 0.06 + d], **kw)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def resolve_run_dir(args):
    """Per-UNet output subdir: <out_dir>/<stage>_<ckpt-stem>/."""
    ckpt_path = pathlib.Path(args.unet_ckpt)
    stage = next((p for p in ckpt_path.parts if re.fullmatch(r"\d+m", p)), None)
    label = f"{stage}_{ckpt_path.stem}" if stage else ckpt_path.stem
    run_dir = pathlib.Path(args.out_dir) / label
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, stage


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
            cond = model._tensor_encode(s2_p, s1_p)
            assert z_0.shape[-2:] == cond.shape[-2:], (
                f"latent/cond spatial mismatch {z_0.shape} vs {cond.shape}"
            )
            # Per-tile noise seed (shard-independent: same tile always gets same eps).
            gi = torch.Generator(device="cpu").manual_seed(args.seed + i)
            eps = torch.randn(z_0.shape, generator=gi).to(device)

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
                psnr = peak_signal_noise_ratio(hr_u8, sr_u8, data_range=255)
                ssim = structural_similarity(hr_u8, sr_u8, channel_axis=2, data_range=255)

                rows.append({
                    "tile": name, "t_frac": frac, "t": t_int,
                    "psnr": float(psnr), "ssim": float(ssim), "mae": float(mae),
                })

    return rows, T


def aggregate_per_t(rows, t_fracs):
    per_t = {}
    for frac in t_fracs:
        sel = [r for r in rows if r["t_frac"] == frac]
        psnr = np.array([r["psnr"] for r in sel])
        ssim = np.array([r["ssim"] for r in sel])
        mae = np.array([r["mae"] for r in sel])
        per_t[frac] = {
            "psnr_mean": float(psnr.mean()), "psnr_std": float(psnr.std()),
            "ssim_mean": float(ssim.mean()), "ssim_std": float(ssim.std()),
            "mae_mean":  float(mae.mean()),  "mae_std":  float(mae.std()),
        }
    return per_t


def write_final_outputs(args, rows, run_dir, t_fracs, T, stage, cmd_line):
    """Write metrics.csv (cmd at top, per-(tile,t) rows, then per-t mean/std rows),
    print the summary, and optionally save the PSNR-vs-t plot."""
    per_t = aggregate_per_t(rows, t_fracs)
    n_tiles = len({r["tile"] for r in rows})

    out_csv = run_dir / "metrics.csv"
    with open(out_csv, "w", newline="") as f:
        f.write(f"# {cmd_line}\n")
        writer = csv.DictWriter(f, fieldnames=["tile", "t_frac", "t", "psnr", "ssim", "mae"])
        writer.writeheader()
        writer.writerows(rows)
        for frac in t_fracs:
            s = per_t[frac]
            writer.writerow({"tile": f"mean_t={frac}", "t_frac": frac, "t": "",
                             "psnr": s["psnr_mean"], "ssim": s["ssim_mean"], "mae": s["mae_mean"]})
            writer.writerow({"tile": f"std_t={frac}",  "t_frac": frac, "t": "",
                             "psnr": s["psnr_std"],  "ssim": s["ssim_std"],  "mae": s["mae_std"]})

    print(f"\n  UNet oracle reconstruction over {n_tiles} tiles  (eps-pred, seed={args.seed})")
    print(f"  {'t/T':>5}  {'t':>4}  {'PSNR':>14}  {'SSIM':>14}  {'MAE':>14}")
    for frac in t_fracs:
        s = per_t[frac]
        t_int = max(0, min(T - 1, int(round(frac * (T - 1)))))
        print(f"  {frac:>5.2f}  {t_int:>4d}  "
              f"{s['psnr_mean']:7.3f} ± {s['psnr_std']:5.3f}  "
              f"{s['ssim_mean']:7.4f} ± {s['ssim_std']:5.4f}  "
              f"{s['mae_mean']:7.4f} ± {s['mae_std']:5.4f}")
    print(f"\nPer-tile-per-t results saved to {out_csv}")

    if args.save_plot:
        plot_path = run_dir / "curve.png"
        title = (f"Kvalitet af {stage} UNet vs Støjniveau (t / T)"
                 if stage else "Kvalitet af UNet vs Støjniveau (t / T)")
        save_curve(per_t, plot_path, title)
        print(f"Plot saved to {plot_path}")


def write_shard_csv(rows, run_dir, shard_idx):
    """Per-shard partial CSV (no command line, no aggregates). Merged by the parent."""
    out = run_dir / f"metrics.shard{shard_idx}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tile", "t_frac", "t", "psnr", "ssim", "mae"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[shard {shard_idx}] wrote {out} ({len(rows)} rows)")


def read_shard_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "tile": r["tile"],
                "t_frac": float(r["t_frac"]),
                "t": int(r["t"]),
                "psnr": float(r["psnr"]),
                "ssim": float(r["ssim"]),
                "mae": float(r["mae"]),
            })
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
    parser.add_argument("--t_fracs", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                        help="Comma-separated fractions of T to test (each in (0,1))")
    parser.add_argument("--out_dir", default=str(ROOT / "test" / "results" / "unet-oracle"))
    parser.add_argument("--max_tiles", type=int, default=None)
    parser.add_argument("--save_plot", action="store_true",
                        help="Save the PSNR-vs-t line plot as curve.png")
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
