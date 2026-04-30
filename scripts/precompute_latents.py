"""Pre-compute VAE latents for all NPZ tiles.

Encodes each tile's aerial (1024×1024) and S2 (128×128 → upsampled) through
the frozen VAE once, saving z_aerial and z_s2 as float16 tensors. UNet training
can then skip the two expensive VAE encode passes per step.

Usage:
    python scripts/precompute_latents.py \
        --npz_dir ~/npz/apr2025/1m-npz \
        --output_dir ~/npz/apr2025/1m-latents \
        --vae_ckpt checkpoints/1m/vae/last.ckpt \
        --config opensr_model/configs/config_1m.yaml \
        --batch_size 8 --device cuda

Output per tile: {output_dir}/{tile_stem}.pt containing:
    z_aerial : (4, 128, 128) float16  — scaled by 0.18215 (matches get_first_stage_encoding)
    z_s2     : (4, 128, 128) float16  — unscaled (matches _build_conditioning)
"""

import argparse
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from opensr_model.autoencoder.autoencoder import AutoencoderKL
from opensr_model.utils import normalize_aerial, normalize_s2

VAE_SCALE_FACTOR = 0.18215

AERIAL_KEYS = ["aerial_r", "aerial_g", "aerial_b", "aerial_nir"]
S2_KEYS     = ["s2_r", "s2_g", "s2_b", "s2_nir"]
S1_KEYS     = ["s1_vv", "s1_vh"]

LR_PAD_SIZE = 128
HR_PAD_SIZE = 1024


def _zero_pad(tensor: torch.Tensor, target: int) -> torch.Tensor:
    _, h, w = tensor.shape
    ph, pw = target - h, target - w
    if ph == 0 and pw == 0:
        return tensor
    return F.pad(tensor, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))


def load_vae(config, ckpt_path: str, device: torch.device) -> AutoencoderKL:
    fsc = dict(config.first_stage_config)
    embed_dim = fsc.pop("embed_dim")
    double_z  = fsc.pop("double_z", True)
    fsc["double_z"] = double_z
    vae = AutoencoderKL(fsc, embed_dim=embed_dim)

    print(f"Loading VAE from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state.items():
        k = k.replace("module.", "")
        if k.startswith("disc.") or k.startswith("lpips_fn."):
            continue
        if k.startswith("vae."):
            k = k[4:]
        cleaned[k] = v
    missing, unexpected = vae.load_state_dict(cleaned, strict=True)
    print(f"  VAE loaded: {len(cleaned)} keys  missing={len(missing)}  unexpected={len(unexpected)}")

    vae = vae.to(device).half().eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def encode_batch(vae, aerial_batch, s2_batch, hr_size, device):
    """Encode one batch of tiles. Returns z_aerial, z_s2 on CPU as float16."""
    dtype = next(vae.parameters()).dtype

    aerial_norm = normalize_aerial(aerial_batch.to(device), stage="norm").to(dtype)
    s2_norm     = normalize_s2(s2_batch.to(device), stage="norm").to(dtype)
    s2_up       = F.interpolate(s2_norm, size=(hr_size, hr_size), mode="bilinear", align_corners=False)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        z_aerial = vae.encode(aerial_norm).mode() * VAE_SCALE_FACTOR
        z_s2     = vae.encode(s2_up).mode()

    return z_aerial.cpu().half(), z_s2.cpu().half()


def collect_tiles(npz_dir: pathlib.Path):
    paths = sorted(npz_dir.rglob("*.npz"))
    ok = []
    for p in paths:
        try:
            with np.load(p, allow_pickle=True) as f:
                keys = set(f.keys())
            if all(k in keys for k in AERIAL_KEYS + S2_KEYS + S1_KEYS):
                ok.append(p)
        except Exception:
            pass
    return ok


def main():
    parser = argparse.ArgumentParser(description="Pre-compute VAE latents for UNet training")
    parser.add_argument("--npz_dir",    type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vae_ckpt",   type=str, required=True)
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device     = torch.device(args.device)
    npz_dir    = pathlib.Path(args.npz_dir).expanduser()
    output_dir = pathlib.Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.config)
    scale_factor  = config.scale_factor                          # 8
    ch_mult       = list(config.first_stage_config.ch_mult)
    vae_downscale = 2 ** (len(ch_mult) - 1)                     # 8
    hr_size       = LR_PAD_SIZE * scale_factor                   # 1024

    vae = load_vae(config, args.vae_ckpt, device)

    print(f"Scanning {npz_dir} …")
    all_paths = collect_tiles(npz_dir)
    pending   = [p for p in all_paths if not (output_dir / f"{p.stem}.pt").exists()]
    n_total, n_pending = len(all_paths), len(pending)

    bytes_per_tile = 2 * 4 * (hr_size // vae_downscale) ** 2 * 2  # 2 tensors × 4ch × H × W × 2 bytes
    print(f"Tiles: {n_total} total, {n_pending} to encode  |  "
          f"Est. output size: {n_total * bytes_per_tile / 1e9:.2f} GB")

    if n_pending == 0:
        print("All tiles already encoded. Nothing to do.")
        return

    bs = args.batch_size
    bar = tqdm(total=n_pending, unit="tile", dynamic_ncols=True)

    def load_batch(paths):
        aerial_list, s2_list = [], []
        for p in paths:
            with np.load(p, allow_pickle=True) as f:
                aerial = torch.from_numpy(
                    np.stack([f[k].astype(np.float32) for k in AERIAL_KEYS])
                )
                s2 = torch.from_numpy(
                    np.stack([f[k].astype(np.float32) for k in S2_KEYS])
                )
            aerial_list.append(_zero_pad(aerial, HR_PAD_SIZE))
            s2_list.append(_zero_pad(s2, LR_PAD_SIZE))
        return torch.stack(aerial_list), torch.stack(s2_list)

    batches = [pending[i : i + bs] for i in range(0, n_pending, bs)]

    with ThreadPoolExecutor(max_workers=2) as pool:
        # Prime the first future
        future = pool.submit(load_batch, batches[0])

        for idx, batch_paths in enumerate(batches):
            # Prefetch next batch while GPU encodes current
            if idx + 1 < len(batches):
                next_future = pool.submit(load_batch, batches[idx + 1])

            aerial_batch, s2_batch = future.result()

            z_aerial_batch, z_s2_batch = encode_batch(vae, aerial_batch, s2_batch, hr_size, device)

            for j, p in enumerate(batch_paths):
                out_path = output_dir / f"{p.stem}.pt"
                torch.save(
                    {"z_aerial": z_aerial_batch[j], "z_s2": z_s2_batch[j]},
                    out_path,
                )

            bar.update(len(batch_paths))
            future = next_future if idx + 1 < len(batches) else None

    bar.close()
    encoded = len(list(output_dir.glob("*.pt")))
    print(f"Done. {encoded}/{n_total} tiles encoded → {output_dir}")


if __name__ == "__main__":
    main()
