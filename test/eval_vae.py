"""Evaluate VAE reconstruction ceiling on aerial GT tiles.

Encodes a clean HR aerial image and decodes it back through the VAE (no diffusion),
then scores the reconstruction against the original. This is the upper bound on what
the latent-diffusion SR pipeline can achieve: any error here is error the diffusion
model can never recover.

Reports mean +/- std of PSNR and SSIM and writes a per-tile CSV, mirroring the
output format of test/eval.py.

IMPORTANT: feed each VAE the spatial size it was TRAINED on, via --pad_size. A VAE is
fully convolutional but its reconstruction degrades badly at unfamiliar input sizes
(the global mid-attention is resolution-sensitive). Tiles are zero-padded (centered) to
--pad_size, reconstructed, then cropped back to native before scoring.
    - 1m VAE: trained at 1024 (1000px tiles padded to 1024)  -> --pad_size 1024
    - 5m VAE: trained at 256  (200px tiles padded to 256)    -> --pad_size 256

Usage:
    # 1m production VAE
    python test/eval_vae.py \
        --vae_ckpt "checkpoints/1m/vae/b4-crop256-gan10/vae-epoch=0023-val_loss=4.804487-brugt-til-unet.ckpt" \
        --npz_dir ~/npz/apr2025/1m-npz --pad_size 1024 --max_tiles 200

    # 5m production VAE
    python test/eval_vae.py \
        --vae_ckpt checkpoints/5m/vae/last.ckpt \
        --npz_dir ~/npz/apr2025/5m-untouched --pad_size 256
"""

import argparse
import csv
import pathlib
import re
import shlex
import sys

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opensr_model.autoencoder.autoencoder import AutoencoderKL
from opensr_model.data import FusionDataset
from opensr_model.utils import normalize_aerial


def infer_vae_config(vae_state: dict) -> dict:
    """Recover the AutoencoderKL ddconfig from a checkpoint's tensor shapes.

    The 1m and 5m production VAEs have different depths (ch_mult [1,2,4,8] vs [1,2,4]),
    so we derive the architecture per checkpoint rather than assuming one config.
    """
    ch, in_ch = vae_state["encoder.conv_in.weight"].shape[:2]
    levels = sorted({int(m.group(1)) for k in vae_state
                     for m in [re.match(r"encoder\.down\.(\d+)\.block\.", k)] if m})
    ch_mult = [vae_state[f"encoder.down.{l}.block.0.conv2.weight"].shape[0] // ch for l in levels]
    num_res_blocks = 1 + max(int(m.group(1)) for k in vae_state
                             for m in [re.match(r"encoder\.down\.0\.block\.(\d+)\.", k)] if m)
    z_channels = vae_state["quant_conv.weight"].shape[1] // 2
    embed_dim = vae_state["quant_conv.weight"].shape[0] // 2
    return dict(
        ddconfig=dict(
            z_channels=z_channels, ch=ch, out_ch=in_ch, ch_mult=ch_mult, resolution=256,
            in_channels=in_ch, double_z=True, num_res_blocks=num_res_blocks,
            attn_resolutions=[], dropout=0.0,
        ),
        embed_dim=embed_dim,
        downsample=2 ** (len(ch_mult) - 1),  # one downsample between each pair of levels
    )


def load_vae(ckpt_path: str, device: str):
    """Build AutoencoderKL (architecture inferred from the ckpt) and load weights.

    Returns (model, downsample_factor).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vae_state = {k.replace("vae.", "", 1): v for k, v in ckpt["state_dict"].items()
                 if k.startswith("vae.")}
    cfg = infer_vae_config(vae_state)
    print(f"  Inferred config: ch_mult={cfg['ddconfig']['ch_mult']} "
          f"num_res_blocks={cfg['ddconfig']['num_res_blocks']} downsample={cfg['downsample']}")

    ae = AutoencoderKL(cfg["ddconfig"], embed_dim=cfg["embed_dim"])
    missing, unexpected = ae.load_state_dict(vae_state, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:3]}{'...' if len(missing) > 3 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
    print(f"  Loaded {len(vae_state)} VAE keys from {ckpt_path}")
    ae.eval().to(device)
    return ae, cfg["downsample"]


def pad_to_multiple(x: torch.Tensor, m: int):
    """Symmetrically zero-pad (B,C,H,W) so H,W are multiples of m. Returns (padded, pad)."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)  # (left, right, top, bottom)
    if ph == 0 and pw == 0:
        return x, pad
    return F.pad(x, pad), pad


def pad_to_size(x: torch.Tensor, size: int):
    """Symmetrically zero-pad (B,C,H,W) to (size, size) — the VAE's training resolution."""
    h, w = x.shape[-2:]
    ph, pw = size - h, size - w
    if ph < 0 or pw < 0:
        raise ValueError(f"--pad_size {size} is smaller than tile {h}x{w}")
    pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)  # (left, right, top, bottom)
    if ph == 0 and pw == 0:
        return x, pad
    return F.pad(x, pad), pad


def crop_pad(x: torch.Tensor, pad) -> torch.Tensor:
    """Undo padding, returning the native-sized center crop."""
    l, r, t, b = pad
    H, W = x.shape[-2:]
    return x[..., t:(H - b) if b else H, l:(W - r) if r else W]


def save_comparison(hr_u8, sr_u8, name, out_dir):
    """Save an 'Original | Reconstruction' RGB side-by-side PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    axes[0].imshow(hr_u8)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(sr_u8)
    axes[1].set_title("Rekonstruktion")
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=120, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="VAE reconstruction-ceiling eval")
    parser.add_argument("--vae_ckpt", required=True, help="Path to VAE .ckpt (Lightning state_dict)")
    parser.add_argument("--npz_dir", required=True, help="Directory of .npz tiles with aerial bands")
    parser.add_argument("--out_dir", default=str(ROOT / "test" / "results" / "vae-recon"),
                        help="Parent dir; a per-VAE subdir with metrics.csv is created here")
    parser.add_argument("--max_tiles", type=int, default=None,
                        help="Evaluate at most this many tiles (deterministic, sorted order)")
    parser.add_argument("--pad_size", type=int, default=None,
                        help="Zero-pad tiles to this size (the VAE's training resolution: "
                             "256 for the 5m VAE, 1024 for the 1m VAE). Strongly recommended — "
                             "feeding a VAE an unfamiliar size badly degrades reconstruction.")
    parser.add_argument("--save_visual", action="store_true",
                        help="Also save an Original|Reconstruction RGB side-by-side PNG per tile")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ae, downsample = load_vae(args.vae_ckpt, device)

    ds = FusionDataset(args.npz_dir, require_aerial=True, pad=False)
    if len(ds) == 0:
        print("No valid tiles found.")
        return
    n = len(ds) if args.max_tiles is None else min(args.max_tiles, len(ds))
    print(f"Evaluating {n} of {len(ds)} tiles...")

    ckpt_path = pathlib.Path(args.vae_ckpt)
    stage = next((p for p in ckpt_path.parts if re.fullmatch(r"\d+m", p)), None)  # e.g. 1m, 5m, 10m
    label = f"{stage}_{ckpt_path.stem}" if stage else ckpt_path.stem
    run_dir = pathlib.Path(args.out_dir) / label
    run_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = run_dir / "visuals"
    if args.save_visual:
        visuals_dir.mkdir(exist_ok=True)

    rows = []
    for i in tqdm(range(n), desc="VAE recon"):
        sample = ds[i]
        name = pathlib.Path(sample["path"]).stem

        aerial = sample["aerial"].unsqueeze(0).to(device)   # (1, 4, H, W), [0, 255]

        # Pad in RAW [0,255] space, THEN normalize. FusionDataset padded tiles with zeros
        # (= black) before normalization during training, so the border must map to -1.
        # Padding after normalization would give a gray (0) border the VAE never saw, which
        # badly degrades the reconstruction.
        if args.pad_size:
            aerial_p, pad = pad_to_size(aerial, args.pad_size)
        else:
            aerial_p, pad = pad_to_multiple(aerial, downsample)
        xp = normalize_aerial(aerial_p, stage="norm")       # -> [-1, 1], border = -1 (black)
        with torch.no_grad():
            recon_p, _ = ae(xp, sample_posterior=False)     # deterministic: posterior mode
        recon = crop_pad(recon_p, pad)                      # native content, [-1, 1]

        x = normalize_aerial(aerial, stage="norm")          # native content, [-1, 1]

        # PSNR/SSIM on RGB [0, 255] (matches test/eval.py)
        hr_u8 = normalize_aerial(x, stage="denorm")[0, :3].cpu().numpy().transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
        sr_u8 = normalize_aerial(recon, stage="denorm")[0, :3].cpu().numpy().transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
        psnr = peak_signal_noise_ratio(hr_u8, sr_u8, data_range=255)
        ssim = structural_similarity(hr_u8, sr_u8, channel_axis=2, data_range=255)

        rows.append({"tile": name, "psnr": float(psnr), "ssim": float(ssim)})

        if args.save_visual:
            save_comparison(hr_u8, sr_u8, name, visuals_dir)

    # Aggregate mean / std once so the CSV summary rows and the stdout print can't drift.
    agg = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in ("psnr", "ssim")}
    mean_row = {"tile": "mean", **{k: float(v.mean()) for k, v in agg.items()}}
    std_row  = {"tile": "std",  **{k: float(v.std())  for k, v in agg.items()}}

    # Save per-tile CSV under a per-VAE subdir
    out_csv = run_dir / "metrics.csv"
    cmd_line = "python " + shlex.join(sys.argv)
    pad_str = args.pad_size if args.pad_size else "auto"
    with open(out_csv, "w", newline="") as f:
        f.write(f"# {cmd_line}\n")
        f.write(f"# effective: pad_size={pad_str} tiles={len(rows)}\n")
        writer = csv.DictWriter(f, fieldnames=["tile", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(mean_row)
        writer.writerow(std_row)

    # Summary
    print(f"\n  VAE reconstruction over {len(rows)} tiles")
    for k, label in [("psnr", "PSNR ↑"), ("ssim", "SSIM ↑")]:
        print(f"  {label:<10}  {mean_row[k]:8.4f} ± {std_row[k]:.4f}")
    print(f"\nPer-tile results saved to {out_csv}")


if __name__ == "__main__":
    main()
