# view_latent_space.py
import argparse
import pathlib
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from opensr_model.autoencoder.autoencoder import AutoencoderKL
from opensr_model.utils import normalize_aerial

parser = argparse.ArgumentParser(description="Visualise VAE latent space for a sample tile")
parser.add_argument("checkpoint", type=str, help="Path to .ckpt file (LightningModule state_dict)")
parser.add_argument("--npz", type=str, default=None,
                    help="NPZ tile to encode. If omitted, samples z ~ N(0,1) from the prior.")
parser.add_argument("--out", type=str, default="latent_space.png")
args = parser.parse_args()

# ── Build and load VAE ────────────────────────────────────────────────────────
ae = AutoencoderKL(dict(z_channels=4, ch=128, out_ch=4, ch_mult=[1,2,4,8],
    resolution=1024, in_channels=4, double_z=True, num_res_blocks=2,
    attn_resolutions=[], dropout=0.0), embed_dim=4)

ckpt = torch.load(args.checkpoint, map_location="cpu")
state = ckpt["state_dict"]
ae.load_state_dict({k.replace("vae.", "", 1): v for k, v in state.items() if k.startswith("vae.")})
ae.eval()

# ── Encode or sample ──────────────────────────────────────────────────────────
with torch.no_grad():
    if args.npz is not None:
        data = np.load(args.npz)
        img = np.stack([data["aerial_r"], data["aerial_g"], data["aerial_b"], data["aerial_nir"]])
        img = torch.from_numpy(img).float().unsqueeze(0)   # (1, 4, 1000, 1000)
        img = F.pad(img, (12, 12, 12, 12))                 # → (1, 4, 1024, 1024)
        x = normalize_aerial(img, stage="norm")            # → [-1, 1]
        posterior = ae.encode(x)
        latent = posterior.mean   # (1, 4, 128, 128)
        mode = "encoded"
    else:
        # Sample z ~ N(0,1) from the prior — shows the decoded latent space
        # without needing an input image.
        latent = torch.randn(1, 4, 128, 128)
        mode = "prior sample"

print(f"Mode: {mode}")
print(f"Latent shape: {latent.shape}")
print(f"Latent range: {latent.min():.3f} to {latent.max():.3f}  "
      f"mean={latent.mean():.3f}  std={latent.std():.3f}")

# ── RGB preview via SD-style linear projection ────────────────────────────────
rgb_factors = torch.tensor([
    [ 0.298,  0.207,  0.208],
    [ 0.187,  0.286,  0.173],
    [-0.158,  0.189,  0.264],
    [-0.184, -0.271, -0.473],
])
preview = torch.einsum('chw,cr->rhw', latent[0].cpu(), rgb_factors)
# Normalise to [0,1] for display
lo, hi = preview.min(), preview.max()
preview_norm = ((preview - lo) / (hi - lo + 1e-8)).clamp(0, 1).permute(1, 2, 0).numpy()

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

channels = ['C0', 'C1', 'C2', 'C3']
for i in range(4):
    ch = latent[0, i].numpy()
    ax = axes[i // 3, i % 3]
    im = ax.imshow(ch, cmap='RdBu_r', vmin=-abs(ch).max(), vmax=abs(ch).max())
    ax.set_title(f'Latent {channels[i]}  [{ch.min():.2f}, {ch.max():.2f}]')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

axes[1, 1].imshow(preview_norm)
axes[1, 1].set_title('RGB preview (SD projection)')
axes[1, 1].axis('off')

# Per-channel histogram in last subplot
ax_hist = axes[1, 2]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for i in range(4):
    vals = latent[0, i].numpy().ravel()
    ax_hist.hist(vals, bins=80, alpha=0.5, color=colors[i], label=f'C{i}')
ax_hist.set_title('Channel value distributions')
ax_hist.legend(fontsize=8)
ax_hist.set_xlabel('Latent value')

ckpt_name = pathlib.Path(args.checkpoint).stem
fig.suptitle(f'Latent space ({mode}) — {ckpt_name}', fontsize=13)
plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"Saved {args.out}")
