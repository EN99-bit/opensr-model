# test_ae.py
import torch, numpy as np, matplotlib
import glob
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from opensr_model.autoencoder.autoencoder import AutoencoderKL
from opensr_model.utils import normalize_aerial

# Load and stack aerial RGBNIR as uint8, pad to 1024x1024 (same as training)
data = np.load("/home/jlund/npz/apr2025/1m-npz/2025_1km_6081_525.npz")
img = np.stack([data["aerial_r"], data["aerial_g"], data["aerial_b"], data["aerial_nir"]])  # (4, 1000, 1000)
img = torch.from_numpy(img).float().unsqueeze(0)  # (1, 4, 1000, 1000)
img = F.pad(img, (12, 12, 12, 12))  # symmetric zero-pad to 1024x1024
x = normalize_aerial(img, stage="norm")  # -> [-1, 1]
print(f"Shape: {x.shape}, range: {x.min():.3f} to {x.max():.3f}")

# Build autoencoder and load weights
ae = AutoencoderKL(dict(z_channels=4, ch=128, out_ch=4, ch_mult=[1,2,4,8],
    resolution=1024, in_channels=4, double_z=True, num_res_blocks=2,
    attn_resolutions=[], dropout=0.0), embed_dim=4)

# matches = glob.glob("/home/jlund/opensr-model/checkpoints/1m/vae/vae-epoch=0005*.ckpt")
# ckpt = torch.load(matches[0], map_location="cpu")
ckpt = torch.load("/home/jlund/opensr-model/checkpoints/1m/vae/last.ckpt", map_location="cpu")
state = ckpt["state_dict"]
# Strip 'vae.' prefix from LightningModule state_dict
ae.load_state_dict({k.replace("vae.", "", 1): v for k, v in state.items() if k.startswith("vae.")})
ae.eval()

# Reconstruct
with torch.no_grad():
    recon, _ = ae(x)

mae = torch.mean(torch.abs(x - recon)).item()
print(f"MAE: {mae:.4f}")

# Denormalize for display
def to_display(t):
    return ((t[0, :3].numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
bands = ['Red', 'Green', 'Blue', 'NIR']
for i in range(4):
    axes[0, i].imshow(((x[0, i].numpy() + 1) / 2).clip(0, 1), cmap='gray')
    axes[0, i].set_title(f'Original {bands[i]}')
    axes[0, i].axis('off')
    axes[1, i].imshow(((recon[0, i].numpy() + 1) / 2).clip(0, 1), cmap='gray')
    axes[1, i].set_title(f'Recon {bands[i]}')
    axes[1, i].axis('off')

axes[2, 0].imshow(to_display(x))
axes[2, 0].set_title('Original RGB')
axes[2, 0].axis('off')
axes[2, 1].imshow(to_display(recon))
axes[2, 1].set_title('Recon RGB')
axes[2, 1].axis('off')
axes[2, 2].imshow(np.abs(to_display(x) - to_display(recon)), cmap='hot')
axes[2, 2].set_title('|Difference|')
axes[2, 2].axis('off')
axes[2, 3].axis('off')

plt.suptitle(f'AE Reconstruction — MAE: {mae:.4f}', fontsize=14)
plt.tight_layout()
plt.savefig('1m-ae_check.png', dpi=150)
print("Saved 1m-ae_check.png")