"""Regenerate the 2x2 report sweep figures (PSNR / LPIPS / Improvement / Hallucination)
from the 3600-tile metrics CSVs, matching the existing thesis figure style.

Usage:
    python test/make_report_plots.py [out_dir]   # default /tmp/report_plots
"""
import csv, io, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = pathlib.Path("/home/jlund/opensr-model/test/results-testset3600")
RUN = RES / "5m_unet-epoch=0098-val_loss=0.102384"
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("/tmp/report_plots")
OUT.mkdir(parents=True, exist_ok=True)

COL  = {"psnr": "#1a1a1a", "lpips": "#e8467c", "improv": "#2ca02c", "halluc": "#ff7f0e"}
MARK = {"psnr": "o",       "lpips": "s",       "improv": "^",       "halluc": "D"}
PANELS = [("psnr", "PSNR ↑ [dB]"), ("lpips", "LPIPS ↓"),
          ("improv", "Improvement ↑"), ("halluc", "Hallucination ↓")]


def read_means(csv_path, prefix, key_field):
    with open(csv_path) as f:
        lines = [l for l in f if not l.startswith("#")]
    rd = csv.DictReader(io.StringIO("".join(lines)))
    rows = []
    for r in rd:
        if r["tile"].startswith(prefix):
            rows.append((float(r[key_field]), {
                "psnr": float(r["psnr"]), "lpips": float(r["lpips"]),
                "halluc": float(r["ha_metric"]), "improv": float(r["im_metric"]),
            }))
    rows.sort()
    return rows


def add_break(ax):
    d = 0.015
    kw = dict(transform=ax.transAxes, color="k", lw=1, clip_on=False)
    ax.plot([-d, +d], [0.03 - d, 0.03 + d], **kw)
    ax.plot([-d, +d], [0.06 - d, 0.06 + d], **kw)


def make_plot(rows, title, xlabel, out_path, logx=False, xticks=None):
    xs = [r[0] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (key, ylab) in zip(axes.flat, PANELS):
        ys = [r[1][key] for r in rows]
        ax.plot(xs, ys, marker=MARK[key], color=COL[key], lw=2, ms=6)
        ax.set_ylabel(ylab)
        ax.grid(True, color="0.85", lw=0.8)
        ax.set_axisbelow(True)
        if logx:
            ax.set_xscale("log")
            ax.set_xticks(xticks or xs)
            ax.set_xticklabels([("%g" % x) for x in (xticks or xs)])
            ax.minorticks_off()
        add_break(ax)
    for ax in axes[1]:
        ax.set_xlabel(xlabel)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


# ── CFG + CFG++ (from the cfg run's metrics.csv) ──────────────────────────────
mcfg = RUN / "metrics.csv"
make_plot(read_means(mcfg, "mean_cfg_gs=", "gs"),
          "5m UNet S1 + S2 guidance scale", "Guidance Scale (gs)", OUT / "cfg_sweep_5m.png")
make_plot(read_means(mcfg, "mean_cfgpp_gs=", "gs"),
          "5m UNet S1 + S2 CFG++", "Guidance Scale (gs)", OUT / "cfgpp_sweep_5m.png")

# ── sampling steps (log x) ────────────────────────────────────────────────────
msteps = RUN / "metrics_steps.csv"
srows = [r for r in read_means(msteps, "mean_steps=", "steps") if r[0] >= 1]
make_plot(srows, "5m UNet S1 + S2 samplingskridt", "Antal samplingskridt (log)",
          OUT / "steps_sweep_5m.png", logx=True, xticks=[1, 2, 5, 10, 20, 50, 100])

# ── oracle (only if the rerun has produced metrics_oracle.csv) ────────────────
moracle = RUN / "metrics_oracle.csv"
if moracle.exists():
    orows = read_means(moracle, "mean_t=", "t_frac")
    make_plot(orows, "5m UNet S1 + S2 denoiser i isolation", "Start t (støjniveau)",
              OUT / "oracle_5m.png")
else:
    print("oracle: metrics_oracle.csv not ready yet — skipped")
