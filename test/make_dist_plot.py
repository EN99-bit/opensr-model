"""Per-tile distribution figure (PSNR + LPIPS) for the 5m S1+S2 model over the
3600-tile test set, in the thesis figure style. Shows spread (std) vs the precise mean.

Usage: python test/make_dist_plot.py [out_dir]
"""
import csv, io, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = pathlib.Path("/home/jlund/opensr-model/test/results-testset3600")
CSV = RES / "e98-val0.102384_eval_steps100_gs1.0" / "metrics.csv"   # S1+S2 gs=1 per-tile
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("/tmp/report_plots")
OUT.mkdir(parents=True, exist_ok=True)

with open(CSV) as f:
    lines = [l for l in f if not l.startswith("#")]
rd = csv.DictReader(io.StringIO("".join(lines)))
psnr, lpips = [], []
for r in rd:
    if r["tile"] in ("mean", "std"):
        continue
    psnr.append(float(r["psnr"])); lpips.append(float(r["lpips"]))
psnr, lpips = np.array(psnr), np.array(lpips)
print(f"n={len(psnr)} tiles")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
panels = [(psnr, "PSNR ↑ [dB]", "#1a1a1a"), (lpips, "LPIPS ↓", "#e8467c")]
for ax, (vals, label, color) in zip(axes, panels):
    m, s = vals.mean(), vals.std()
    ax.hist(vals, bins=45, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axvspan(m - s, m + s, color=color, alpha=0.12)
    ax.axvline(m, color="k", lw=1.6, ls="--")
    ax.set_xlabel(label)
    ax.set_ylabel("Antal tiles")
    ax.grid(True, color="0.85", lw=0.8, axis="y")
    ax.set_axisbelow(True)
    ax.text(0.97, 0.95, f"Gns. {m:.2f}\nStd. {s:.2f}", transform=ax.transAxes,
            ha="right", va="top", bbox=dict(boxstyle="round", fc="white", ec="0.7"))
fig.suptitle("5m UNet S1 + S2 fordeling over testsættet", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = OUT / "per_tile_dist_5m.png"
fig.savefig(out, dpi=100, bbox_inches="tight")
print("wrote", out)
