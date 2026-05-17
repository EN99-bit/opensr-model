"""Pre-compute VAE latents for 5m→1m cascade UNet training.

Encodes each paired tile's 1m aerial (target) and 5m aerial (conditioning) through
the frozen 1m VAE once, saving z_aerial and z_aerial_5m as float16 tensors.

    z_aerial    : 1m aerial (1024×1024) encoded × VAE_SCALE_FACTOR  — target latent
    z_aerial_5m : 5m aerial (256×256) upsampled to 1024, encoded    — conditioning latent

Auto-detects all available CUDA GPUs and distributes tiles across them.
S1 is cheap to normalize on-the-fly during training and is NOT pre-computed.

Usage:
    python scripts/precompute_latents_5to1m.py \\
        --npz_dir_5m ~/npz/apr2025/5m-npz \\
        --npz_dir_1m ~/npz/apr2025/1m-npz \\
        --output_dir ~/npz/apr2025/5m-to-1m-latents \\
        --vae_ckpt checkpoints/1m/vae/b4-crop256-gan10/vae-epoch=0023-val_loss=4.804487-brugt-til-unet.ckpt \\
        --config opensr_model/configs/config_1m.yaml \\
        --batch_size 2 --device cuda
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

VAE_SCALE_FACTOR   = 0.18215
AERIAL_KEYS        = ["aerial_r", "aerial_g", "aerial_b", "aerial_nir"]
S2_KEYS            = ["s2_r", "s2_g", "s2_b", "s2_nir"]
HR_PAD_SIZE        = 1024
AERIAL_5M_PAD_SIZE = 256
LR_PAD_SIZE        = 128


def _zero_pad(tensor: torch.Tensor, target: int) -> torch.Tensor:
    _, h, w = tensor.shape
    ph, pw = target - h, target - w
    if ph == 0 and pw == 0:
        return tensor
    return F.pad(tensor, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))


def load_vae(config, ckpt_path: str, device: torch.device) -> AutoencoderKL:
    fsc = dict(config.first_stage_config)
    embed_dim = fsc.pop("embed_dim")
    fsc["double_z"] = fsc.pop("double_z", True)
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


def encode_batch(vae, aerial_1m_batch, aerial_5m_batch, s2_batch, device):
    dtype = next(vae.parameters()).dtype

    aerial_norm = normalize_aerial(aerial_1m_batch.to(device), stage="norm").to(dtype)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        z_aerial = vae.encode(aerial_norm).mode() * VAE_SCALE_FACTOR

    aerial_5m_norm = normalize_aerial(aerial_5m_batch.to(device), stage="norm").to(dtype)
    aerial_5m_up   = F.interpolate(aerial_5m_norm, size=(HR_PAD_SIZE, HR_PAD_SIZE),
                                   mode="bilinear", align_corners=False)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        z_aerial_5m = vae.encode(aerial_5m_up).mode()

    s2_norm = normalize_s2(s2_batch.to(device), stage="norm").to(dtype)
    s2_up   = F.interpolate(s2_norm, size=(HR_PAD_SIZE, HR_PAD_SIZE),
                            mode="bilinear", align_corners=False)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        z_s2 = vae.encode(s2_up).mode()

    return z_aerial.cpu().half(), z_aerial_5m.cpu().half(), z_s2.cpu().half()


def find_pairs(npz_dir_5m: pathlib.Path, npz_dir_1m: pathlib.Path):
    files_5m = {p.stem: p for p in sorted(npz_dir_5m.glob("*.npz"))}
    files_1m = {p.stem: p for p in sorted(npz_dir_1m.glob("*.npz"))}
    common = sorted(files_5m.keys() & files_1m.keys())
    return [(files_5m[k], files_1m[k]) for k in common]


def worker(rank, world_size, pending, args_dict):
    device = torch.device(f"cuda:{rank}" if "cuda" in args_dict["device"] else args_dict["device"])
    output_dir = pathlib.Path(args_dict["output_dir"])
    config = OmegaConf.load(args_dict["config"])

    vae = load_vae(config, args_dict["vae_ckpt"], device)

    shard = pending[rank::world_size]
    if not shard:
        return

    bs = args_dict["batch_size"]
    batches = [shard[i:i + bs] for i in range(0, len(shard), bs)]

    def load_batch(pairs):
        aerial_1m_list, aerial_5m_list, s2_list = [], [], []
        for p5, p1 in pairs:
            with np.load(p1, allow_pickle=True) as f1:
                aerial_1m = torch.from_numpy(
                    np.stack([f1[k].astype(np.float32) for k in AERIAL_KEYS])
                )
            with np.load(p5, allow_pickle=True) as f5:
                aerial_5m = torch.from_numpy(
                    np.stack([f5[k].astype(np.float32) for k in AERIAL_KEYS])
                )
                s2 = torch.from_numpy(
                    np.stack([f5[k].astype(np.float32) for k in S2_KEYS])
                )
            aerial_1m_list.append(_zero_pad(aerial_1m, HR_PAD_SIZE))
            aerial_5m_list.append(_zero_pad(aerial_5m, AERIAL_5M_PAD_SIZE))
            s2_list.append(_zero_pad(s2, LR_PAD_SIZE))
        return torch.stack(aerial_1m_list), torch.stack(aerial_5m_list), torch.stack(s2_list)

    bar = tqdm(total=len(shard), desc=f"GPU {rank}", position=rank, leave=True, unit="tile")

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(load_batch, batches[0])
        for idx, batch_pairs in enumerate(batches):
            if idx + 1 < len(batches):
                next_future = pool.submit(load_batch, batches[idx + 1])
            aerial_1m_batch, aerial_5m_batch, s2_batch = future.result()
            z_aerial_batch, z_aerial_5m_batch, z_s2_batch = encode_batch(
                vae, aerial_1m_batch, aerial_5m_batch, s2_batch, device
            )
            for j, (p5, _) in enumerate(batch_pairs):
                torch.save(
                    {"z_aerial":    z_aerial_batch[j].clone(),
                     "z_aerial_5m": z_aerial_5m_batch[j].clone(),
                     "z_s2":        z_s2_batch[j].clone()},
                    output_dir / f"{p5.stem}.pt",
                )
            bar.update(len(batch_pairs))
            future = next_future if idx + 1 < len(batches) else None

    bar.close()


def main():
    parser = argparse.ArgumentParser(description="Pre-compute VAE latents for 5m→1m UNet training")
    parser.add_argument("--npz_dir_5m",  type=str, required=True)
    parser.add_argument("--npz_dir_1m",  type=str, required=True)
    parser.add_argument("--output_dir",  type=str, required=True)
    parser.add_argument("--vae_ckpt",    type=str, required=True)
    parser.add_argument("--config",      type=str, required=True)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = find_pairs(
        pathlib.Path(args.npz_dir_5m).expanduser(),
        pathlib.Path(args.npz_dir_1m).expanduser(),
    )
    pending = [(p5, p1) for p5, p1 in all_pairs
               if not (output_dir / f"{p5.stem}.pt").exists()]

    config = OmegaConf.load(args.config)
    ch_mult       = list(config.first_stage_config.ch_mult)
    vae_downscale = 2 ** (len(ch_mult) - 1)
    latent_size   = HR_PAD_SIZE // vae_downscale
    bytes_per_tile = 2 * 4 * latent_size ** 2 * 2

    print(f"5m tiles: {sum(1 for p in pathlib.Path(args.npz_dir_5m).expanduser().glob('*.npz'))}  |  "
          f"1m tiles: {sum(1 for p in pathlib.Path(args.npz_dir_1m).expanduser().glob('*.npz'))}  |  "
          f"paired: {len(all_pairs)}")
    print(f"Tiles to encode: {len(pending)}/{len(all_pairs)}  |  "
          f"Est. output size: {len(all_pairs) * bytes_per_tile / 1e9:.2f} GB")

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
    print(f"\nDone. {encoded}/{len(all_pairs)} tiles encoded → {output_dir}")


if __name__ == "__main__":
    main()
