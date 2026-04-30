"""One-off script: replace random VAE weights in a UNet checkpoint with correct ones.

Usage:
    python scripts/patch_vae_in_unet_ckpt.py \
        --unet_ckpt checkpoints/1m/unet/last.ckpt \
        --vae_ckpt  checkpoints/1m/vae/last.ckpt \
        --out       checkpoints/1m/unet/last-vae-patched.ckpt
"""

import argparse
import pathlib
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_ckpt", required=True)
    parser.add_argument("--vae_ckpt",  required=True)
    parser.add_argument("--out",       required=True)
    args = parser.parse_args()

    print(f"Loading UNet checkpoint: {args.unet_ckpt}")
    unet = torch.load(args.unet_ckpt, map_location="cpu", weights_only=False)

    print(f"Loading VAE checkpoint:  {args.vae_ckpt}")
    vae_raw = torch.load(args.vae_ckpt, map_location="cpu", weights_only=False)
    vae_sd  = vae_raw.get("state_dict", vae_raw)

    # VAE checkpoint keys look like "vae.encoder.*" → strip "vae." prefix
    # UNet checkpoint keys look like "ldm.first_stage_model.encoder.*"
    # Map: "ldm.first_stage_model." + bare_key  ←→  "vae." + bare_key
    replaced = 0
    missing  = []
    for unet_key in list(unet["state_dict"].keys()):
        if not unet_key.startswith("ldm.first_stage_model."):
            continue
        bare    = unet_key[len("ldm.first_stage_model."):]
        vae_key = "vae." + bare
        if vae_key in vae_sd:
            unet["state_dict"][unet_key] = vae_sd[vae_key]
            replaced += 1
        else:
            missing.append(bare)

    print(f"Replaced {replaced} VAE keys")
    if missing:
        print(f"WARNING: {len(missing)} keys not found in VAE checkpoint: {missing[:5]}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(unet, out)
    print(f"Saved patched checkpoint → {out}")


if __name__ == "__main__":
    main()
