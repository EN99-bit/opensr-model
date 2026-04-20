import torch
import numpy as np
from omegaconf import OmegaConf
from opensr_model.srmodel import SRLatentDiffusion
from opensr_model.data import FusionDataset
from PIL import Image
import pathlib

TEST_DIR = pathlib.Path(__file__).parent
NPZ_FILE = TEST_DIR / "2025_1km_6240_485.npz"

# 1. Build model (random weights)
cfg = OmegaConf.load(TEST_DIR.parent / "opensr_model" / "configs" / "config_10m.yaml")
print("Building model...")
model = SRLatentDiffusion(cfg, device="cpu")
print("Model ready.")

# 2. Load the test NPZ tile
ds = FusionDataset(root=TEST_DIR, file_list=[str(NPZ_FILE)])
sample = ds[0]
s1 = sample["s1"].unsqueeze(0)   # (1, 2, 128, 128)
s2 = sample["s2"].unsqueeze(0)   # (1, 4, 128, 128)
print(f"Input: s1={s1.shape}, s2={s2.shape}")
print(f"File: {sample['path']}")

# 3. Run inference (10 DDIM steps - fast on CPU)
print("Running DDIM sampling (10 steps)...")
sr = model.forward(s2, s1, sampling_steps=10, histogram_matching=False)
print(f"Output SR: {sr.shape}, min={sr.min().item():.1f}, max={sr.max().item():.1f}")

# 4. Save SR output RGB preview
sr_rgb = sr[0, :3].clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
Image.fromarray(sr_rgb).save(TEST_DIR / "output_test_1_sr.png")
print("Saved output_test_1_sr.png")

# 5. Save aerial ground truth for comparison
aerial_rgb = sample["aerial"][:3].clamp(0, 255).byte().numpy().transpose(1, 2, 0)
Image.fromarray(aerial_rgb).save(TEST_DIR / "output_test_1_aerial_gt.png")
print("Saved output_test_1_aerial_gt.png")

# 6. Save S2 input RGB preview
s2_rgb = s2[0, :3]
s2_rgb = (s2_rgb / s2_rgb.max() * 255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
Image.fromarray(s2_rgb).save(TEST_DIR / "output_test_1_s2_input.png")
print("Saved output_test_1_s2_input.png")

print("Done! Check test/ folder for output_test_1_*.png files")
