"""Augment a folder of NPZ tiles with flips and 90° rotations (8x multiplier).

For each input tile, produces 8 output files:
  original, hflip, vflip, rot90, rot180, rot270, hflip+rot90, vflip+rot90

All bands (s1, s2, aerial) receive the identical spatial transform so
alignment is preserved. S1/S2 and aerial may have different spatial sizes —
each is transformed independently.

Usage:
    python augment_npz.py --in_dir ~/npz/apr2025/5m-npz --out_dir ~/npz/apr2025/5m-npz-aug
    python augment_npz.py --in_dir data --out_dir data_aug --workers 8
"""

import argparse
import pathlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# ── 8 dihedral transforms (D4 symmetry group) ─────────────────────────────────

TRANSFORMS = [
    ("orig",    lambda x: x),
    ("hflip",   lambda x: np.fliplr(x)),
    ("vflip",   lambda x: np.flipud(x)),
    ("rot90",   lambda x: np.rot90(x, k=1)),
    ("rot180",  lambda x: np.rot90(x, k=2)),
    ("rot270",  lambda x: np.rot90(x, k=3)),
    ("hf_r90",  lambda x: np.rot90(np.fliplr(x), k=1)),
    ("vf_r90",  lambda x: np.rot90(np.flipud(x), k=1)),
]

ALL_KEYS = ["s1_vv", "s1_vh", "s2_b", "s2_g", "s2_r", "s2_nir",
            "aerial_r", "aerial_g", "aerial_b", "aerial_nir"]


def augment_file(npz_path: pathlib.Path, out_dir: pathlib.Path):
    """Load one NPZ, apply all 8 transforms, save to out_dir."""
    with np.load(npz_path, allow_pickle=True) as npz:
        arrays = {k: npz[k] for k in npz.files}

    stem = npz_path.stem
    saved = 0

    for suffix, fn in TRANSFORMS:
        out_path = out_dir / f"{stem}_{suffix}.npz"
        augmented = {}
        for k, v in arrays.items():
            if k in ALL_KEYS and v.ndim == 2:
                augmented[k] = np.ascontiguousarray(fn(v))
            else:
                augmented[k] = v  # pass non-spatial keys through unchanged
        np.savez_compressed(out_path, **augmented)
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(description="8x augment NPZ tiles with flips+rotations")
    parser.add_argument("--in_dir",  type=str, required=True, help="Input folder containing .npz files")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder for augmented .npz files")
    parser.add_argument("--workers", type=int, default=4,    help="Parallel worker processes")
    args = parser.parse_args()

    in_dir  = pathlib.Path(args.in_dir).expanduser()
    out_dir = pathlib.Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.rglob("*.npz"))
    if not files:
        print(f"No .npz files found in {in_dir}")
        return

    print(f"Found {len(files)} files → {len(files) * 8} output tiles")
    print(f"Output: {out_dir}")

    total_saved = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(augment_file, f, out_dir): f for f in files}
        with tqdm(total=len(files), unit="tile") as bar:
            for future in as_completed(futures):
                total_saved += future.result()
                bar.update(1)

    print(f"Done. {total_saved} files written to {out_dir}")


if __name__ == "__main__":
    main()
