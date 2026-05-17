"""Cascade SR uncertainty estimation: S2+S1 (10m) → 5m aerial → 1m aerial.

Runs the full cascade pipeline N times with different random noise (z_T) each time.
All GPUs collaborate on each tile: GPU i handles runs i, i+4, i+8, ... (interleaved).
Per-pixel std across N runs is overlaid as a magma heatmap on the mean SR output.

Output per tile:
  {stem}.png        — panel: S2 | S1 | Stage1 5m SR + uncertainty | Mean 1m SR + uncertainty | GT
  {stem}_std_5m.npy — raw (4, 256, 256) float32 std tensor for stage 1
  {stem}_std_1m.npy — raw (4, 1024, 1024) float32 std tensor for stage 2

Usage:
    python test/test-inference-cascade-uncertainty.py \\
        --input_dir ~/npz/apr2025/1m-untouched \\
        --unet_ckpt_5m checkpoints/5m/unet/last.ckpt \\
        --unet_ckpt_1m checkpoints/5to1m/unet/last.ckpt \\
        --n_runs 20
"""

import argparse
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opensr_model.srmodel import SRLatentDiffusion
from opensr_model.utils import normalize_aerial, normalize_s1, normalize_s2

AERIAL_KEYS      = ["aerial_r", "aerial_g", "aerial_b", "aerial_nir"]
S2_KEYS          = ["s2_r", "s2_g", "s2_b", "s2_nir"]
S1_KEYS          = ["s1_vv", "s1_vh"]

LR_PAD           = 128
LR_NATIVE        = 100
AERIAL_5M_PAD    = 256
AERIAL_1M_PAD    = 1024
AERIAL_1M_NATIVE = 1000
LATENT_1M        = 128


def _zero_pad(tensor: torch.Tensor, target: int) -> torch.Tensor:
    _, h, w = tensor.shape
    ph, pw = target - h, target - w
    if ph == 0 and pw == 0:
        return tensor
    return F.pad(tensor, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))


def load_trained_weights(model: SRLatentDiffusion, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    remapped = {k[len("ldm."):]: v for k, v in state_dict.items() if k.startswith("ldm.")}
    missing, unexpected = model.model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print(f"  Loaded {len(remapped)} keys from {ckpt_path}")


def tensor_to_rgb(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 4:
        t = t[0]
    img = t[:3].cpu().float().numpy()
    return np.transpose(np.clip(img, 0, 255).astype(np.uint8), (1, 2, 0))


def tensor_to_s1_rgb(t: torch.Tensor) -> np.ndarray:
    vv = t[0, 0].cpu().numpy()
    vh = t[0, 1].cpu().numpy()

    def stretch(a):
        lo, hi = np.percentile(a, 2), np.percentile(a, 98)
        return (np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)

    return np.stack([stretch(vv), stretch(vh), stretch(vv)], axis=-1)


@torch.no_grad()
def run_stage2(
    model2: SRLatentDiffusion,
    sr_5m: torch.Tensor,
    s1_128: torch.Tensor,
    s2_128: torch.Tensor,
    device: str,
    steps: int,
    guidance: float,
    cfg_plus_plus: bool = False,
) -> torch.Tensor:
    dtype = next(model2.model.first_stage_model.parameters()).dtype

    _, _, h, w = sr_5m.shape
    if h != AERIAL_5M_PAD or w != AERIAL_5M_PAD:
        sr_5m = F.interpolate(sr_5m, (AERIAL_5M_PAD, AERIAL_5M_PAD),
                              mode="bilinear", align_corners=False)

    a_norm = normalize_aerial(sr_5m.to(device), stage="norm").to(dtype)
    a_up   = F.interpolate(a_norm, (AERIAL_1M_PAD, AERIAL_1M_PAD),
                           mode="bilinear", align_corners=False)
    z_5m   = model2.model.first_stage_model.encode(a_up).mode()

    s1_norm = normalize_s1(s1_128.to(device), stage="norm").to(dtype)
    s1_up   = F.interpolate(s1_norm, (LATENT_1M, LATENT_1M),
                            mode="bilinear", align_corners=False)

    s2_norm = normalize_s2(s2_128.to(device), stage="norm").to(dtype)
    s2_up   = F.interpolate(s2_norm, (AERIAL_1M_PAD, AERIAL_1M_PAD),
                            mode="bilinear", align_corners=False)
    z_s2    = model2.model.first_stage_model.encode(s2_up).mode()

    conditioning      = torch.cat([z_5m, s1_up, z_s2], dim=1)
    null_conditioning = torch.zeros_like(conditioning)

    ddim, latent, time_range = model2._prepare_model(conditioning, custom_steps=steps)

    for i, step in enumerate(time_range):
        index = steps - i - 1
        t = torch.full((1,), step, device=device, dtype=torch.long)
        if cfg_plus_plus or guidance > 1.0:
            e_uncond = model2.model.apply_model(latent, t, cond=null_conditioning)
            e_cond   = model2.model.apply_model(latent, t, cond=conditioning)
            if cfg_plus_plus:
                latent = model2._ddim_step_cfg_pp(latent, e_cond, e_uncond, guidance, index, ddim, 1.0)
            else:
                e_t    = e_uncond + guidance * (e_cond - e_uncond)
                latent = model2._ddim_step(latent, e_t, index, ddim, 1.0)
        else:
            outs = ddim.p_sample_ddim(x=latent, c=conditioning, t=step, index=index,
                                      use_original_steps=False, temperature=1.0)
            latent, _ = outs

    return model2._tensor_decode(latent, spe_cor=False)


def overlay_uncertainty(mean_img: np.ndarray, unc_norm: np.ndarray, alpha_scale: float = 0.6) -> np.ndarray:
    alpha   = unc_norm[:, :, None] * alpha_scale
    heatmap = plt.cm.magma(unc_norm)[:, :, :3]
    base    = mean_img.astype(np.float32) / 255.0
    return (((1 - alpha) * base + alpha * heatmap).clip(0, 1) * 255).astype(np.uint8)


def worker(rank: int, n_gpus: int, args, tile_paths: list, run_dir: pathlib.Path,
           shared_5m: torch.Tensor, shared_1m: torch.Tensor, barrier):
    import os
    device = f"cuda:{rank}"

    # Silence noisy model-init prints from non-primary workers
    if rank != 0:
        sys.stdout = open(os.devnull, "w")

    cfg1 = OmegaConf.load(ROOT / "opensr_model" / "configs" / "config_10m.yaml")
    cfg2 = OmegaConf.load(ROOT / "opensr_model" / "configs" / "config_5m_to_1m_with_s2.yaml")

    print(f"[{device}] Loading stage 1 ...")
    model1 = SRLatentDiffusion(cfg1, device=device)
    load_trained_weights(model1, args.unet_ckpt_5m)
    model1.eval()

    if args.zero_s2 or args.zero_s1:
        _orig_encode = model1._tensor_encode
        if args.zero_s2 and args.zero_s1:
            def _patched_encode(X_s2, X_s1):
                cond = _orig_encode(X_s2, X_s1); cond[:] = 0; return cond
        elif args.zero_s2:
            def _patched_encode(X_s2, X_s1):
                cond = _orig_encode(X_s2, X_s1); cond[:, :4] = 0; return cond
        else:
            def _patched_encode(X_s2, X_s1):
                cond = _orig_encode(X_s2, X_s1); cond[:, 4:] = 0; return cond
        model1._tensor_encode = _patched_encode

    print(f"[{device}] Loading stage 2 ...")
    model2 = SRLatentDiffusion(cfg2, device=device)
    load_trained_weights(model2, args.unet_ckpt_1m)
    model2.eval()

    lrp     = (LR_PAD - LR_NATIVE) // 2
    hrp     = (AERIAL_1M_PAD - AERIAL_1M_NATIVE) // 2
    display = AERIAL_1M_NATIVE
    _p5     = lrp * 2

    my_run_count = len(range(rank, args.n_runs, n_gpus))
    tile_bar = tqdm(tile_paths, desc="tiles    ", position=0,         leave=True)  if rank == 0 else tile_paths
    run_bar  = tqdm(total=my_run_count, desc=f"GPU {rank} runs", position=rank + 1, leave=True)

    for npz_path in tile_bar:
        stem = npz_path.stem

        with np.load(npz_path, allow_pickle=True) as f:
            s2_raw = torch.from_numpy(np.stack([f[k].astype(np.float32) for k in S2_KEYS]))
            s1_raw = torch.from_numpy(np.stack([f[k].astype(np.float32) for k in S1_KEYS]))
            gt_raw = torch.from_numpy(np.stack([f[k].astype(np.float32) for k in AERIAL_KEYS]))

        s2_pad  = _zero_pad(s2_raw, LR_PAD).unsqueeze(0)
        s1_pad  = _zero_pad(s1_raw, LR_PAD).unsqueeze(0)
        s2_disp = torch.zeros_like(s2_pad) if args.zero_s2 else s2_pad
        s1_disp = torch.zeros_like(s1_pad) if args.zero_s1 else s1_pad

        run_bar.reset()
        # Each GPU handles every n_gpus-th run (interleaved)
        for run_idx in range(rank, args.n_runs, n_gpus):
            with torch.no_grad():
                sr_5m_i = model1.forward(
                    s2_pad.to(device), s1_pad.to(device),
                    sampling_steps=args.steps, guidance_scale=args.guidance,
                    cfg_plus_plus=args.cfg_plus_plus,
                    histogram_matching=False, apply_nodata_mask=False,
                ).cpu()
            sr_1m_i = run_stage2(model2, sr_5m_i, s1_disp, s2_pad, device,
                                 args.steps, args.guidance, args.cfg_plus_plus).cpu()
            shared_5m[run_idx].copy_(sr_5m_i[0])
            shared_1m[run_idx].copy_(sr_1m_i[0])
            run_bar.update(1)

        barrier.wait()  # all GPUs done with this tile's runs

        if rank == 0:
            stack_5m = shared_5m[:args.n_runs]           # (N, 4, 256, 256)
            stack_1m = shared_1m[:args.n_runs]           # (N, 4, 1024, 1024)
            std_5m   = stack_5m.std(dim=0)               # (4, 256, 256)
            std_1m   = stack_1m.std(dim=0)               # (4, 1024, 1024)
            rep_5m   = stack_5m[0].unsqueeze(0)          # (1, 4, 256, 256)  — single representative run
            rep_1m   = stack_1m[0].unsqueeze(0)          # (1, 4, 1024, 1024)

            np.save(run_dir / f"{stem}_std_5m.npy", std_5m.numpy())
            np.save(run_dir / f"{stem}_std_1m.npy", std_1m.numpy())

            # 1m uncertainty
            unc_crop = std_1m[:3].mean(dim=0)[hrp:hrp + display, hrp:hrp + display].numpy()
            unc_norm = (unc_crop - unc_crop.min()) / (unc_crop.max() + 1e-6)

            # 5m uncertainty
            unc_5m_crop = std_5m[:3].mean(dim=0)[_p5:AERIAL_5M_PAD - _p5, _p5:AERIAL_5M_PAD - _p5]
            unc_5m_up   = F.interpolate(unc_5m_crop[None, None].float(), (display, display),
                                        mode="bilinear", align_corners=False).squeeze().numpy()
            unc_5m_norm = (unc_5m_up - unc_5m_up.min()) / (unc_5m_up.max() + 1e-6)

            s2_native = s2_disp[:, :, lrp:lrp + LR_NATIVE, lrp:lrp + LR_NATIVE]
            s1_native = s1_disp[:, :, lrp:lrp + LR_NATIVE, lrp:lrp + LR_NATIVE]

            s2_up  = F.interpolate(s2_native, (display, display), mode="bilinear", align_corners=False)
            s2_img = tensor_to_rgb((s2_up / s2_up[:, :3].max().clamp(min=1e-6) * 255).clamp(0, 255))

            s1_up  = F.interpolate(s1_native, (display, display), mode="bilinear", align_corners=False)
            s1_img = tensor_to_s1_rgb(s1_up)

            sr_5m_native = rep_5m[:, :, _p5:AERIAL_5M_PAD - _p5, _p5:AERIAL_5M_PAD - _p5]
            stg1_up      = F.interpolate(sr_5m_native, (display, display), mode="bilinear", align_corners=False)
            stg1_img     = overlay_uncertainty(tensor_to_rgb(stg1_up), unc_5m_norm)

            rep_crop    = rep_1m[:, :, hrp:hrp + display, hrp:hrp + display]
            overlay_img = overlay_uncertainty(tensor_to_rgb(rep_crop), unc_norm)

            gt_pad  = _zero_pad(gt_raw, AERIAL_1M_PAD).unsqueeze(0)
            gt_img  = tensor_to_rgb(gt_pad[:, :, hrp:hrp + display, hrp:hrp + display])

            panel = np.concatenate([s2_img, s1_img, stg1_img, overlay_img, gt_img], axis=1)
            Image.fromarray(panel).save(run_dir / f"{stem}.png")

        barrier.wait()  # rank 0 done saving; all workers move to next tile


def main():
    parser = argparse.ArgumentParser(
        description="Cascade SR uncertainty estimation: S2+S1 → 5m → 1m (N runs, multi-GPU)"
    )
    parser.add_argument("--input_dir",     type=str, required=True)
    parser.add_argument("--unet_ckpt_5m",  type=str, default=None)
    parser.add_argument("--unet_ckpt_1m",  type=str, default=None)
    parser.add_argument("--n_runs",        type=int,   default=20)
    parser.add_argument("--steps",         type=int,   default=100)
    parser.add_argument("--guidance",      type=float, default=1.0)
    parser.add_argument("--cfg_plus_plus", action="store_true", default=False)
    parser.add_argument("--zero_s1",       action="store_true", default=False)
    parser.add_argument("--zero_s2",       action="store_true", default=False)
    parser.add_argument("--out_dir",       type=str,   default=str(ROOT / "test" / "results"))
    args = parser.parse_args()

    if args.unet_ckpt_5m is None:
        args.unet_ckpt_5m = str(ROOT / "checkpoints" / "5m" / "unet" / "last.ckpt")
    if args.unet_ckpt_1m is None:
        args.unet_ckpt_1m = str(ROOT / "checkpoints" / "5to1m" / "unet" / "last.ckpt")

    stem1 = pathlib.Path(args.unet_ckpt_5m).stem
    stem2 = pathlib.Path(args.unet_ckpt_1m).stem
    m1    = re.search(r'epoch=(\d+)', stem1)
    m2    = re.search(r'epoch=(\d+)', stem2)
    label = (f"uncertainty_5m-e{int(m1.group(1)) if m1 else 'x'}"
             f"_1m-e{int(m2.group(1)) if m2 else 'x'}"
             f"_g{args.guidance:g}"
             f"_n{args.n_runs}"
             f"{'_cpp' if args.cfg_plus_plus else ''}"
             f"{'_nos1' if args.zero_s1 else ''}"
             f"{'_nos2' if args.zero_s2 else ''}")
    run_dir = pathlib.Path(args.out_dir) / label
    run_dir.mkdir(parents=True, exist_ok=True)

    tile_paths = sorted(pathlib.Path(args.input_dir).expanduser().glob("*.npz"))
    if not tile_paths:
        print("No .npz tiles found. Exiting.")
        return

    n_gpus  = torch.cuda.device_count() or 1
    print(f"GPUs: {n_gpus}  |  Tiles: {len(tile_paths)}  |  Runs per tile: {args.n_runs}")
    print(f"Output: {run_dir}")

    # Shared CPU tensors written by workers, read by rank 0
    shared_5m = torch.zeros(args.n_runs, 4, AERIAL_5M_PAD, AERIAL_5M_PAD).share_memory_()
    shared_1m = torch.zeros(args.n_runs, 4, AERIAL_1M_PAD, AERIAL_1M_PAD).share_memory_()
    barrier   = mp.get_context("spawn").Barrier(n_gpus)

    if n_gpus == 1:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        worker(0, 1, args, tile_paths, run_dir, shared_5m, shared_1m, barrier)
    else:
        mp.spawn(worker,
                 args=(n_gpus, args, tile_paths, run_dir, shared_5m, shared_1m, barrier),
                 nprocs=n_gpus, join=True)

    print(f"\nDone! {len(tile_paths)} tiles saved to {run_dir}/")


if __name__ == "__main__":
    main()
