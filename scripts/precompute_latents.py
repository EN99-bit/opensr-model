"""Pre-compute VAE latents for all NPZ tiles.

Encodes each tile's aerial and S2 (128×128 → upsampled to hr_size) through
the frozen VAE once, saving z_aerial and z_s2 as float16 tensors. UNet training
can then skip the two expensive VAE encode passes per step.

Auto-detects all available CUDA GPUs and distributes tiles across them.

Usage:
    python scripts/precompute_latents.py \
        --npz_dir ~/npz/apr2025/1m-npz \
        --output_dir ~/npz/apr2025/1m-latents \
        --vae_ckpt checkpoints/1m/vae/last.ckpt \
        --config opensr_model/configs/config_1m.yaml \
        --batch_size 8 --device cuda

Output per tile: {output_dir}/{tile_stem}.pt containing:
    aerial_mean   : (4, 128, 128) float16  — unscaled VAE posterior mean for aerial
    aerial_logvar : (4, 128, 128) float16  — unscaled VAE posterior log-variance for aerial
    s2_mean       : (4, 128, 128) float16  — unscaled VAE posterior mean for S2
    s2_logvar     : (4, 128, 128) float16  — unscaled VAE posterior log-variance for S2

Sampling z ~ N(mean, exp(0.5*logvar)) happens at dataloader time, recovering the
stochastic regularization of the on-the-fly VAE path.
"""

import argparse
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.multiprocessing as mp
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
    vae.load_state_dict(cleaned, strict=True)
    return vae.to(device).half().eval().requires_grad_(False)


def encode_batch(vae, aerial_batch, s2_batch, hr_size, device):
    dtype = next(vae.parameters()).dtype

    aerial_norm = normalize_aerial(aerial_batch.to(device), stage="norm").to(dtype)
    s2_norm     = normalize_s2(s2_batch.to(device), stage="norm").to(dtype)
    s2_up       = F.interpolate(s2_norm, size=(hr_size, hr_size), mode="bilinear", align_corners=False)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        aerial_post = vae.encode(aerial_norm)
        s2_post     = vae.encode(s2_up)

    return (aerial_post.mean.cpu().half(), aerial_post.logvar.cpu().half(),
            s2_post.mean.cpu().half(),     s2_post.logvar.cpu().half())


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


def worker(rank, world_size, pending, args_dict):
    device = torch.device(f"cuda:{rank}" if "cuda" in args_dict["device"] else args_dict["device"])
    output_dir = pathlib.Path(args_dict["output_dir"])
    config = OmegaConf.load(args_dict["config"])

    ch_mult   = list(config.first_stage_config.ch_mult)
    vae_downscale = 2 ** (len(ch_mult) - 1)
    scale_factor  = config.scale_factor
    hr_size       = LR_PAD_SIZE * scale_factor

    vae = load_vae(config, args_dict["vae_ckpt"], device)

    shard = pending[rank::world_size]
    if not shard:
        return

    bs = args_dict["batch_size"]
    batches = [shard[i:i + bs] for i in range(0, len(shard), bs)]

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
            aerial_list.append(_zero_pad(aerial, hr_size))
            s2_list.append(_zero_pad(s2, LR_PAD_SIZE))
        return torch.stack(aerial_list), torch.stack(s2_list)

    bar = tqdm(total=len(shard), desc=f"GPU {rank}", position=rank, leave=True, unit="tile")

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(load_batch, batches[0])
        for idx, batch_paths in enumerate(batches):
            if idx + 1 < len(batches):
                next_future = pool.submit(load_batch, batches[idx + 1])
            aerial_batch, s2_batch = future.result()
            aerial_mean, aerial_logvar, s2_mean, s2_logvar = encode_batch(vae, aerial_batch, s2_batch, hr_size, device)
            for j, p in enumerate(batch_paths):
                torch.save(
                    {"aerial_mean":   aerial_mean[j].clone(),
                     "aerial_logvar": aerial_logvar[j].clone(),
                     "s2_mean":       s2_mean[j].clone(),
                     "s2_logvar":     s2_logvar[j].clone()},
                    output_dir / f"{p.stem}.pt",
                )
            bar.update(len(batch_paths))
            future = next_future if idx + 1 < len(batches) else None

    bar.close()


def main():
    parser = argparse.ArgumentParser(description="Pre-compute VAE latents for UNet training")
    parser.add_argument("--npz_dir",    type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vae_ckpt",   type=str, required=True)
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    npz_dir    = pathlib.Path(args.npz_dir).expanduser()
    output_dir = pathlib.Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.config)
    ch_mult       = list(config.first_stage_config.ch_mult)
    vae_downscale = 2 ** (len(ch_mult) - 1)
    scale_factor  = config.scale_factor
    hr_size       = LR_PAD_SIZE * scale_factor

    print(f"Scanning {npz_dir} ...")
    all_paths = collect_tiles(npz_dir)
    pending   = [p for p in all_paths if not (output_dir / f"{p.stem}.pt").exists()]

    bytes_per_tile = 2 * 4 * (hr_size // vae_downscale) ** 2 * 2
    print(f"Tiles: {len(all_paths)} total, {len(pending)} to encode  |  "
          f"Est. output size: {len(all_paths) * bytes_per_tile / 1e9:.2f} GB")

    if not pending:
        print("All tiles already encoded. Nothing to do.")
        return

    num_gpus = torch.cuda.device_count() if "cuda" in args.device else 1
    args_dict = {
        "device":     args.device,
        "output_dir": str(output_dir),
        "vae_ckpt":   args.vae_ckpt,
        "config":     args.config,
        "batch_size": args.batch_size,
    }

    if num_gpus > 1:
        print(f"Spawning {num_gpus} workers (one per GPU)...")
        mp.spawn(worker, args=(num_gpus, pending, args_dict), nprocs=num_gpus, join=True)
    else:
        worker(0, 1, pending, args_dict)

    encoded = len(list(output_dir.glob("*.pt")))
    print(f"\nDone. {encoded}/{len(all_paths)} tiles encoded → {output_dir}")


if __name__ == "__main__":
    main()
