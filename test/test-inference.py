"""Inference test using locally trained VAE + UNet checkpoints.

Loads the UNet Lightning checkpoint (which contains both frozen VAE weights
and trained UNet weights under the 'ldm.*' prefix) into SRLatentDiffusion,
then runs inference on the test NPZ tile and saves comparison images.

Usage:
    python test/test-inference.py
    python test/test-inference.py --unet_ckpt checkpoints/unet/best.ckpt
"""

import argparse
import pathlib
import re
import sys
import torch
from omegaconf import OmegaConf
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opensr_model.srmodel import SRLatentDiffusion
from opensr_model.data import FusionDataset

TEST_DIR = pathlib.Path(__file__).parent
DEFAULT_UNET_CKPT = ROOT / "checkpoints" / "unet" / "last.ckpt"


def load_trained_weights(model: SRLatentDiffusion, unet_ckpt: str):
    """Load trained weights from a LitUNetDenoiser Lightning checkpoint.

    The checkpoint stores all weights under the 'ldm.*' prefix (self.ldm in
    LitUNetDenoiser). SRLatentDiffusion.model is the LatentDiffusion directly,
    so we strip that prefix. The checkpoint includes both the frozen VAE and
    trained UNet, so no separate VAE checkpoint is needed.
    """
    ckpt = torch.load(unet_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    remapped = {}
    for k, v in state_dict.items():
        if k.startswith("ldm."):
            remapped[k[len("ldm."):]] = v

    missing, unexpected = model.model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print(f"  Loaded {len(remapped)} keys from {unet_ckpt}")


def main():
    parser = argparse.ArgumentParser(description="Inference with locally trained VAE+UNet")
    parser.add_argument("--unet_ckpt", type=str, default=str(DEFAULT_UNET_CKPT),
                        help="Path to LitUNetDenoiser Lightning checkpoint")
    parser.add_argument("--sampling_steps", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="CFG guidance scale (1.0 = disabled)") #use ~6.0 for best results
    parser.add_argument("--out_dir", type=str, default=str(TEST_DIR))
    parser.add_argument("--npz", type=str, default=str(TEST_DIR / "2025_1km_6096_725-ny.npz"),
                        help="Path to input NPZ tile")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    npz_file = pathlib.Path(args.npz)
    npz_stem = npz_file.stem
    ckpt_stem = pathlib.Path(args.unet_ckpt).stem
    m = re.search(r'epoch=(\d+).*val_loss=([\d.]+)', ckpt_stem)
    short_ckpt = f"e{int(m.group(1))}-val{m.group(2)}" if m else ckpt_stem
    run_name = f"{short_ckpt}_steps{args.sampling_steps}_gs{args.guidance_scale}_{npz_stem}"
    out_dir = pathlib.Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Build model
    cfg = OmegaConf.load(ROOT / "opensr_model" / "configs" / "config_10m.yaml")
    print("Building SRLatentDiffusion...")
    model = SRLatentDiffusion(cfg, device=device)

    # Load trained weights
    print(f"Loading checkpoint: {args.unet_ckpt}")
    load_trained_weights(model, args.unet_ckpt)
    model.eval()

    # Load test tile
    ds = FusionDataset(root=TEST_DIR, file_list=[str(npz_file)])
    sample = ds[0]
    s1 = sample["s1"].unsqueeze(0).to(device)    # (1, 2, 128, 128)
    s2 = sample["s2"].unsqueeze(0).to(device)    # (1, 4, 128, 128)
    print(f"Input: s1={s1.shape}, s2={s2.shape}")

    # Sanity check: verify UNet produces non-degenerate output
    with torch.no_grad():
        dummy_z = torch.randn(1, 4, 64, 64).to(device)
        dummy_t = torch.tensor([500]).to(device)
        dummy_cond = torch.randn(1, 6, 64, 64).to(device)
        dummy_out = model.model.apply_model(dummy_z, dummy_t, cond=dummy_cond)
        print(f"UNet sanity check: mean={dummy_out.mean():.4f}, std={dummy_out.std():.4f}")

    # Run inference
    print(f"Running DDIM sampling ({args.sampling_steps} steps)...")
    with torch.no_grad():
        sr = model.forward(s2, s1, sampling_steps=args.sampling_steps, histogram_matching=False, guidance_scale=args.guidance_scale)
    print(f"Output SR: {sr.shape}, min={sr.min().item():.1f}, max={sr.max().item():.1f}")

    # Save SR output RGB preview
    sr_rgb = sr[0, :3].clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
    Image.fromarray(sr_rgb).save(out_dir / "inference_sr.png")
    print("Saved inference_sr.png")

    # Save aerial ground truth for comparison
    aerial_rgb = sample["aerial"][:3].clamp(0, 255).byte().numpy().transpose(1, 2, 0)
    Image.fromarray(aerial_rgb).save(out_dir / "inference_aerial_gt.png")
    print("Saved inference_aerial_gt.png")

    # Save S2 input RGB preview
    s2_rgb = s2[0, :3]
    s2_rgb = (s2_rgb / s2_rgb.max() * 255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
    Image.fromarray(s2_rgb).save(out_dir / "inference_s2_input.png")
    print("Saved inference_s2_input.png")

    print("Done! Check test/ folder for inference_*.png files")


if __name__ == "__main__":
    main()
