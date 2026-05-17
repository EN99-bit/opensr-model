"""Dataset for loading S1+S2+Aerial NPZ tiles produced by the Bachelor pipeline.

Each .npz file contains per-band arrays:
    s1_vv       : (H, W)   float32  — Sentinel-1 VV in dB
    s1_vh       : (H, W)   float32  — Sentinel-1 VH in dB
    s2_b        : (H, W)   uint16   — Sentinel-2 B02 (Blue)
    s2_g        : (H, W)   uint16   — Sentinel-2 B03 (Green)
    s2_r        : (H, W)   uint16   — Sentinel-2 B04 (Red)
    s2_nir      : (H, W)   uint16   — Sentinel-2 B08 (NIR)
    aerial_r    : (H2, W2) uint8    — Aerial Red
    aerial_g    : (H2, W2) uint8    — Aerial Green
    aerial_b    : (H2, W2) uint8    — Aerial Blue
    aerial_nir  : (H2, W2) uint8    — Aerial NIR

Current tile sizes (5 m aerial, 1 km x 1 km tiles):
    S1/S2  : 100 x 100  (10 m)  ->  padded to 128 x 128
    Aerial : 200 x 200  ( 5 m)  ->  padded to 256 x 256

Usage:
    from opensr_model.data import FusionDataset
    from torch.utils.data import DataLoader

    ds = FusionDataset("path/to/npz_tiles")
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)

    for batch in dl:
        s1, s2, aerial = batch["s1"], batch["s2"], batch["aerial"]
"""

import os
import pathlib
from typing import Union, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from opensr_model.utils import normalize_s1

# Native (unpadded) tile sizes
LR_NATIVE = 100    # S1/S2 native pixels per tile
HR_NATIVE = 1000   # Aerial native pixels per tile

# Padded output sizes (must be divisible by 2^num_downsamples for UNet)
LR_PAD_SIZE = 128   # S1/S2 padded spatial size
HR_PAD_SIZE = 1024  # Aerial padded spatial size (1000×1000 → 1024×1024)

# Number of zero-padded pixels on each side in padded space
LR_PAD = (LR_PAD_SIZE - LR_NATIVE) // 2   # 14
HR_PAD = (HR_PAD_SIZE - HR_NATIVE) // 2   # 12


class FusionDataset(Dataset):
    """PyTorch Dataset that reads S1+S2+Aerial NPZ tiles with per-band keys.

    Stacks individual band arrays into channel-first tensors and
    zero-pads to UNet-friendly sizes (128x128 LR, 256x256 HR).

    Args:
        root:            Path to directory containing .npz files (searched recursively).
        file_list:       Optional explicit list of .npz paths (overrides root scan).
        require_aerial:  If True (default), skip tiles that lack aerial bands.
        pad:             If True (default), zero-pad to LR_PAD_SIZE / HR_PAD_SIZE.
    """

    # Expected per-band keys in the NPZ files
    S1_KEYS = ["s1_vv", "s1_vh"]
    S2_KEYS = ["s2_r", "s2_g", "s2_b", "s2_nir"]
    AERIAL_KEYS = ["aerial_r", "aerial_g", "aerial_b", "aerial_nir"]

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        file_list: Optional[List[str]] = None,
        require_aerial: bool = True,
        pad: bool = True,
    ):
        super().__init__()
        self.require_aerial = require_aerial
        self.pad = pad

        # Collect .npz paths
        if file_list is not None:
            all_paths = [pathlib.Path(p) for p in file_list]
        else:
            root = pathlib.Path(root)
            all_paths = sorted(root.rglob("*.npz"))

        # Filter tiles
        self.paths: List[pathlib.Path] = []
        for p in all_paths:
            if self._tile_ok(p):
                self.paths.append(p)

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[FusionDataset] {len(self.paths)}/{len(all_paths)} tiles accepted "
                  f"(require_aerial={require_aerial}, pad={pad})")

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _tile_ok(self, path: pathlib.Path) -> bool:
        """Quick check: does the tile contain the required bands?"""
        try:
            with np.load(path, allow_pickle=True) as npz:
                keys = set(npz.keys())
                # Must have S1 + S2 bands
                if not all(k in keys for k in self.S1_KEYS + self.S2_KEYS):
                    return False
                # Optionally require aerial bands
                if self.require_aerial:
                    if not all(k in keys for k in self.AERIAL_KEYS):
                        return False
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Padding helper
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_pad(tensor: torch.Tensor, target_size: int) -> torch.Tensor:
        """Zero-pad a (C, H, W) tensor to (C, target_size, target_size)."""
        _, h, w = tensor.shape
        pad_h = target_size - h
        pad_w = target_size - w
        if pad_h == 0 and pad_w == 0:
            return tensor
        # F.pad format: (left, right, top, bottom)
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        return F.pad(tensor, padding, mode="constant", value=0)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        """Load a single tile, stack bands, and optionally pad.

        Returns dict with keys:
            s1      : (2, LR_PAD_SIZE, LR_PAD_SIZE)   float32  — dB values
            s2      : (4, LR_PAD_SIZE, LR_PAD_SIZE)   float32  — DN values (uint16->float32)
            aerial  : (4, HR_PAD_SIZE, HR_PAD_SIZE)   float32  — pixel values 0-255
            path    : str                              — file path for debugging
        """
        path = self.paths[idx]
        with np.load(path, allow_pickle=True) as npz:
            # Stack S1 bands -> (2, H, W)
            s1 = torch.from_numpy(
                np.stack([npz[k].astype(np.float32) for k in self.S1_KEYS], axis=0)
            )
            # Stack S2 bands -> (4, H, W)
            s2 = torch.from_numpy(
                np.stack([npz[k].astype(np.float32) for k in self.S2_KEYS], axis=0)
            )
            # Stack aerial bands -> (4, H2, W2) or zeros if missing
            if all(k in npz for k in self.AERIAL_KEYS):
                aerial = torch.from_numpy(
                    np.stack([npz[k].astype(np.float32) for k in self.AERIAL_KEYS], axis=0)
                )
            else:
                hr = s2.shape[-1] * 2  # fallback: assume 2x scale
                aerial = torch.zeros(4, hr, hr)

        # Zero-pad to UNet-friendly sizes
        if self.pad:
            s1 = self._zero_pad(s1, LR_PAD_SIZE)
            s2 = self._zero_pad(s2, LR_PAD_SIZE)
            aerial = self._zero_pad(aerial, HR_PAD_SIZE)

        return {
            "s1": s1,
            "s2": s2,
            "aerial": aerial,
            "path": str(path),
        }


class LatentFusionDataset(Dataset):
    """Loads pre-computed VAE latents + raw S1 for UNet training.

    Expects a directory of .pt files (one per tile) produced by
    scripts/precompute_latents.py, each containing:
        z_aerial : (4, 128, 128) float16
        z_s2     : (4, 128, 128) float16

    S1 is loaded from the original NPZ and normalized on-the-fly (cheap).

    Args:
        root:       Path to directory containing original .npz files.
        latent_dir: Path to directory containing pre-computed .pt files.
        file_list:  Optional explicit list of .npz paths (overrides root scan).
    """

    S1_KEYS = ["s1_vv", "s1_vh"]

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        latent_dir: Union[str, pathlib.Path],
        file_list: Optional[List[str]] = None,
    ):
        super().__init__()
        self.latent_dir = pathlib.Path(latent_dir)

        if file_list is not None:
            all_paths = [pathlib.Path(p) for p in file_list]
        else:
            root = pathlib.Path(root)
            all_paths = sorted(root.rglob("*.npz"))

        self.paths: List[pathlib.Path] = []
        for p in all_paths:
            if self._tile_ok(p):
                self.paths.append(p)

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[LatentFusionDataset] {len(self.paths)}/{len(all_paths)} tiles accepted "
                  f"(latent_dir={self.latent_dir})")

    def _tile_ok(self, path: pathlib.Path) -> bool:
        if not (self.latent_dir / f"{path.stem}.pt").exists():
            return False
        try:
            with np.load(path, allow_pickle=True) as npz:
                return all(k in npz for k in self.S1_KEYS)
        except Exception:
            return False

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        lat = torch.load(
            self.latent_dir / f"{path.stem}.pt",
            map_location="cpu",
            weights_only=True,
        )
        with np.load(path, allow_pickle=True) as npz:
            s1 = torch.from_numpy(
                np.stack([npz[k].astype(np.float32) for k in self.S1_KEYS])
            )  # (2, 100, 100) raw

        s1 = FusionDataset._zero_pad(s1, LR_PAD_SIZE)  # → (2, 128, 128)
        s1_cond = normalize_s1(s1, stage="norm")

        if "aerial_mean" in lat:
            # New format: sample z ~ N(mean, exp(0.5*logvar)) — recovers on-the-fly stochasticity
            VAE_SCALE = 0.18215
            z_aerial = (lat["aerial_mean"].float() + torch.exp(0.5 * lat["aerial_logvar"].float()) * torch.randn(lat["aerial_mean"].shape)) * VAE_SCALE
            z_s2     = lat["s2_mean"].float() + torch.exp(0.5 * lat["s2_logvar"].float()) * torch.randn(lat["s2_mean"].shape)
        else:
            # Backward compat: old files store precomputed mode
            z_aerial = lat["z_aerial"]
            z_s2     = lat["z_s2"]

        latent_size = z_aerial.shape[-1]
        if s1_cond.shape[-1] != latent_size:
            s1_cond = F.interpolate(
                s1_cond.unsqueeze(0), size=(latent_size, latent_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        return {
            "z_aerial": z_aerial,  # (4, latent_size, latent_size)
            "z_s2":     z_s2,      # (4, latent_size, latent_size)
            "s1_cond":  s1_cond,   # (2, latent_size, latent_size) float32
            "path":     str(path),
        }


# ------------------------------------------------------------------
# Convenience: train/val split
# ------------------------------------------------------------------

def make_train_val_datasets(
    root: Union[str, pathlib.Path],
    val_frac: float = 0.1,
    seed: int = 42,
    require_aerial: bool = True,
    pad: bool = True,
) -> tuple:
    """Create train and val FusionDatasets from a single directory.

    Splits the file list deterministically based on `seed`.

    Returns:
        (train_dataset, val_dataset)
    """
    root = pathlib.Path(root)
    all_paths = sorted(root.rglob("*.npz"))

    # Shuffle deterministically
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_paths))
    n_val = max(1, int(len(all_paths) * val_frac))

    val_paths = [str(all_paths[i]) for i in indices[:n_val]]
    train_paths = [str(all_paths[i]) for i in indices[n_val:]]

    train_ds = FusionDataset(
        root=root,
        file_list=train_paths,
        require_aerial=require_aerial,
        pad=pad,
    )
    val_ds = FusionDataset(
        root=root,
        file_list=val_paths,
        require_aerial=require_aerial,
        pad=pad,
    )

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"[Split] Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_ds, val_ds
