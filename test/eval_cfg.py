"""CFG (and CFG++) impact diagnostic.

Sweep classifier-free guidance scale at full DDIM sampling and score with the
same opensr_test metrics test/eval.py uses, then plot the headline tradeoff.

Per tile, per guidance value, per CFG mode (standard / CFG++), runs:
    sr = model.forward(s2, s1,
                       sampling_steps=...,
                       guidance_scale=gs,
                       cfg_plus_plus=cfg_pp,
                       histogram_matching=False)
and scores with `opensr_test.Metrics().compute(lr=, sr=, hr=)`.

Output:
    metrics.csv   — per (tile, gs, cfg_mode) rows + aggregates at the bottom
    curve.png     — 2x2 panel plot (PSNR / SSIM / Reflektans / Syntese)
                    each panel: two lines (CFG vs CFG++) vs guidance scale

Multi-GPU: same subprocess sharding as eval_unet.py — tiles are striped across
N child processes pinned to one GPU each, then merged.

Usage:
    # 1m
    python test/eval_cfg.py \
        --unet_ckpt checkpoints/1m/unet/last.ckpt \
        --npz_dir ~/npz/apr2025/1m-npz --pad_size 1024 \
        --max_tiles 20 --sampling_steps 50 --save_plot

    # 5m
    python test/eval_cfg.py \
        --unet_ckpt checkpoints/5m/unet-latents/last.ckpt \
        --config opensr_model/configs/config_10m.yaml \
        --npz_dir ~/npz/apr2025/5m-untouched --pad_size 256 \
        --sampling_steps 50 --save_plot
"""

import argparse
import csv
import os
import pathlib
import re
import shlex
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tqdm import tqdm

try:
    import opensr_test
except ImportError:
    print("opensr-test not installed. Run: pip install opensr-test")
    sys.exit(1)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opensr_model.srmodel import SRLatentDiffusion
from opensr_model.data import FusionDataset, LR_PAD_SIZE
from opensr_model.utils import normalize_s2

from eval import load_trained_weights


# opensr_test produces these in addition to our PSNR/SSIM.
OPENSR_METRICS = ["reflectance", "spectral", "spatial", "synthesis",
                  "ha_metric", "om_metric", "im_metric"]
METRIC_COLS = ["psnr", "ssim", *OPENSR_METRICS]

# Sentinel panel key: render om_metric + ha_metric together on one shared y-axis.
OM_HA_PANEL = "om_ha_panel"
HA_COLOR = "tab:blue"
OM_COLOR = "tab:orange"

# Plot panels (subset chosen to tell the CFG tradeoff story).
PANEL_METRICS_FULL = [
    ("psnr",        "PSNR ↑"),
    ("ssim",        "SSIM ↑"),
    ("im_metric",   "Improvement ↑"),
    (OM_HA_PANEL,   "Omission vs. Hallucination"),
]
# Used when opensr_test is skipped (e.g. on the 1m stage where its 10× geometry breaks).
PANEL_METRICS_PIXEL = [
    ("psnr", "PSNR ↑"),
    ("ssim", "SSIM ↑"),
]


def pad_to_size(x: torch.Tensor, size: int):
    h, w = x.shape[-2:]
    ph, pw = size - h, size - w
    if ph < 0 or pw < 0:
        raise ValueError(f"pad size {size} smaller than tile {h}x{w}")
    pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
    return (F.pad(x, pad), pad) if (ph or pw) else (x, pad)


def to_rgb_u8_norm(t01):
    """(C,H,W) tensor in [0,1] -> RGB uint8 (H,W,3) for skimage metrics."""
    arr = t01[:3].cpu().numpy().transpose(1, 2, 0) * 255.0
    return arr.clip(0, 255).astype(np.uint8)


def score_batch(model, samples, args, gs, cfg_pp):
    """Batched DDIM at (gs, cfg_pp) for B tiles.

    Returns `(results, sr_thumbs)`:
      - `results`: list of per-tile metric dicts.
      - `sr_thumbs`: list of (H, W, 3) uint8 RGB arrays — the same SR data the metric
        functions already use, returned for optional saving (effectively free).

    Stacking tiles along the batch dim lets the UNet process them all in one forward
    per DDIM step — much higher GPU utilization than batch_size=1.
    Assumes all tiles in `samples` share the same native HR (true for one dataset).
    """
    device = next(model.parameters()).device
    s2 = torch.stack([s["s2"] for s in samples]).to(device)
    s1 = torch.stack([s["s1"] for s in samples]).to(device)
    aerial = torch.stack([s["aerial"] for s in samples]).to(device).float()  # [0, 255]
    native_hr = aerial.shape[-1]

    s2_p, _ = pad_to_size(s2, LR_PAD_SIZE)
    s1_p, _ = pad_to_size(s1, LR_PAD_SIZE)

    # Seed right before each forward so every (gs, cfg_pp) draws the SAME initial
    # latent (and same per-step noise) — isolating the guidance rule as the only
    # variable when comparing CFG vs CFG++.
    if args.seed is not None and args.seed >= 0:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    with torch.no_grad():
        sr = model.forward(
            s2_p, s1_p,
            sampling_steps=args.sampling_steps,
            sampling_eta=args.eta,            # None -> forward falls back to the config value
            guidance_scale=gs,
            cfg_plus_plus=cfg_pp,
            histogram_matching=False,
            apply_nodata_mask=False,
        )  # [0, 255], (B, 4, sr_H, sr_W)

    sr_size = sr.shape[-1]
    p = (sr_size - native_hr) // 2
    sr_native = sr[..., p:p + native_hr, p:p + native_hr]

    results = []
    sr_thumbs = []
    for b in range(len(samples)):
        lr_norm = (normalize_s2(s2[b].float(), stage="norm") + 1.0) / 2.0
        sr_norm = (sr_native[b] / 255.0).clamp(0, 1)
        hr_norm = (aerial[b] / 255.0).clamp(0, 1)

        out = {}
        if getattr(args, "no_opensr_test", False):
            # Skip the opensr_test pass; fill its columns with NaN so the CSV schema is stable.
            for k in OPENSR_METRICS:
                out[k] = float("nan")
        else:
            try:
                # opensr_test (satalign) breaks on scale factors > 4. Upsample LR if necessary.
                lr_eval = lr_norm
                if hr_norm.shape[-1] / lr_norm.shape[-1] > 4:
                    target_size = hr_norm.shape[-1] // 4
                    lr_eval = F.interpolate(
                        lr_norm.unsqueeze(0), size=(target_size, target_size), mode="bilinear", align_corners=False
                    )[0]

                r = opensr_test.Metrics().compute(lr=lr_eval, sr=sr_norm, hr=hr_norm)
                for k in OPENSR_METRICS:
                    out[k] = float(r.get(k, float("nan")))
            except Exception as e:
                print(f"  opensr_test failed (gs={gs}, cfg++={cfg_pp}): {e}")
                for k in OPENSR_METRICS:
                    out[k] = float("nan")

        hr_u8 = to_rgb_u8_norm(hr_norm)
        sr_u8 = to_rgb_u8_norm(sr_norm)
        out["psnr"] = float(peak_signal_noise_ratio(hr_u8, sr_u8, data_range=255))
        out["ssim"] = float(structural_similarity(hr_u8, sr_u8, channel_axis=2, data_range=255))
        results.append(out)
        sr_thumbs.append(sr_u8)
    return results, sr_thumbs


def resolve_run_dir(args):
    """Per-UNet output subdir: <out_dir>/<raw-stage>_<ckpt-stem>/.

    Returns (run_dir, display_stage). Folder keeps the literal path component
    for uniqueness; display_stage normalizes cascade names ("5to1m..." → "1m")
    for the figure title.
    """
    ckpt_path = pathlib.Path(args.unet_ckpt)
    raw_stage = next(
        (p for p in ckpt_path.parts
         if re.fullmatch(r"\d+m", p) or re.match(r"^\d+to\d+m", p)),
        None,
    )
    label = f"{raw_stage}_{ckpt_path.stem}" if raw_stage else ckpt_path.stem
    run_dir = pathlib.Path(args.out_dir) / label
    run_dir.mkdir(parents=True, exist_ok=True)

    display_stage = raw_stage
    if raw_stage:
        m = re.match(r"^\d+to(\d+)m", raw_stage)
        if m:
            display_stage = f"{m.group(1)}m"
    return run_dir, display_stage


def save_tile_grid(tile_name, hr_rgb, gs_to_sr_rgb, mode_label, out_path, label_fmt="gs={:g}"):
    """Square-ish grid: Original + SR@gs_1, SR@gs_2, … laid out near-square.

    Picks `ncols = ceil(sqrt(n_panels))` and `nrows = ceil(n / ncols)`, so for the
    default 8-value sweep (= 9 panels total) you get a clean 3×3.
    """
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gs_sorted = sorted(gs_to_sr_rgb.keys())
    n_panels = 1 + len(gs_sorted)
    ncols = math.ceil(math.sqrt(n_panels))
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3.3),
                             squeeze=False)
    axes_flat = list(axes.flat)

    axes_flat[0].imshow(hr_rgb)
    axes_flat[0].set_title("Original")
    axes_flat[0].axis("off")
    for i, gs in enumerate(gs_sorted):
        axes_flat[i + 1].imshow(gs_to_sr_rgb[gs])
        axes_flat[i + 1].set_title(label_fmt.format(gs))
        axes_flat[i + 1].axis("off")
    # Hide any leftover cells (when n_panels < nrows × ncols).
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(f"{tile_name} ({mode_label})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def run_eval_loop(args, sweeps, shard_idx=0, shard_count=1):
    """`sweeps` is a list of (cfg_pp_bool, gs_values_list). Each tile is evaluated
    against every (cfg_pp, gs) pair in the union of all sweeps."""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"[shard {shard_idx}/{shard_count}]" if shard_count > 1 else ""
    print(f"{tag} Device: {device}")

    cfg = OmegaConf.load(args.config)
    model = SRLatentDiffusion(cfg, device=device)
    print(f"{tag} Loading UNet (+ bundled VAE): {args.unet_ckpt}")
    load_trained_weights(model, args.unet_ckpt)
    model = model.to(device).eval()
    print(f"{tag} sampling_steps={args.sampling_steps}, sweeps={sweeps}")

    ds = FusionDataset(args.npz_dir, require_aerial=True, pad=False)
    if len(ds) == 0:
        print("No valid tiles found.")
        return []
    n_total = len(ds) if args.max_tiles is None else min(args.max_tiles, len(ds))
    tile_indices = list(range(shard_idx, n_total, shard_count))
    n_per_tile = sum(len(gs_list) for _, gs_list in sweeps)

    bs = max(1, args.batch_size)
    tile_batches = [tile_indices[i:i + bs] for i in range(0, len(tile_indices), bs)]
    print(f"{tag} Evaluating {len(tile_indices)} tiles in {len(tile_batches)} batches "
          f"of up to {bs} × {n_per_tile} runs each")

    save_images = getattr(args, "save_images", False)
    if save_images:
        images_dir, _ = resolve_run_dir(args)
        images_dir = images_dir / "images"
        images_dir.mkdir(exist_ok=True)

    show_progress = (shard_idx == 0)
    rows = []
    iterator = tqdm(tile_batches, desc=f"CFG sweep {tag}".strip(),
                    disable=not show_progress)
    for batch_indices in iterator:
        samples = [ds[i] for i in batch_indices]
        names = [pathlib.Path(s["path"]).stem for s in samples]
        # When --save_images is set, accumulate SR thumbs per (tile, mode) so we can
        # render one grid (Original + SR@each gs) per (tile, mode) at end of batch.
        tile_srs = {} if save_images else None

        for cfg_pp, gs_list in sweeps:
            for gs in gs_list:
                batch_results, batch_thumbs = score_batch(model, samples, args, gs, cfg_pp)
                for name, m in zip(names, batch_results):
                    rows.append({
                        "tile": name, "gs": gs, "cfg_pp": int(cfg_pp),
                        **{k: m[k] for k in METRIC_COLS},
                    })
                if tile_srs is not None:
                    for name, thumb in zip(names, batch_thumbs):
                        tile_srs.setdefault((name, int(cfg_pp)), {})[gs] = thumb

        if tile_srs is not None:
            for sample in samples:
                name = pathlib.Path(sample["path"]).stem
                hr_rgb = (sample["aerial"][:3].numpy().transpose(1, 2, 0)
                          .clip(0, 255).astype(np.uint8))
                for cfg_pp, _ in sweeps:
                    cp = int(cfg_pp)
                    key = (name, cp)
                    if key not in tile_srs:
                        continue
                    mode_label = "CFG++" if cp == 1 else "CFG"
                    mode_tag = "cfgpp" if cp == 1 else "cfg"
                    out_path = images_dir / f"{name}_{mode_tag}.png"
                    save_tile_grid(name, hr_rgb, tile_srs[key], mode_label, out_path)

    return rows


def aggregate(rows, sweeps):
    """Returns {cfg_pp_int: {gs: {metric: (mean, std)}}} for each (cfg_pp, gs_list) in sweeps."""
    out = {}
    for cfg_pp, gs_list in sweeps:
        cp = int(cfg_pp)
        out[cp] = {g: {} for g in gs_list}
        for gs in gs_list:
            sel = [r for r in rows if int(r["cfg_pp"]) == cp and float(r["gs"]) == gs]
            for k in METRIC_COLS:
                v = np.array([r[k] for r in sel], dtype=np.float64)
                v = v[~np.isnan(v)]
                if len(v) == 0:
                    out[cp][gs][k] = (float("nan"), float("nan"))
                else:
                    out[cp][gs][k] = (float(v.mean()), float(v.std()))
    return out


def _draw_om_ha_panel(ax, agg_mode, gs_list, x_label,
                      baseline_agg=None, baseline_x=None, baseline_ref_label="CFG, gs=1"):
    """Render omission + hallucination as two lines on one shared y-axis.

    Both are ↓-better and live in the same numeric range, so a single axis
    keeps their magnitudes directly comparable. If `baseline_x` is present in
    `baseline_agg`, draw the no-guidance reference (CFG gs=1) for each metric as
    a dashed line in that metric's colour. Headroom on top leaves room for the
    legend.
    """
    ha = [agg_mode[g]["ha_metric"][0] for g in gs_list]
    om = [agg_mode[g]["om_metric"][0] for g in gs_list]

    l_ha, = ax.plot(gs_list, ha, marker="o", lw=2, color=HA_COLOR, label="Hallucination ↓")
    l_om, = ax.plot(gs_list, om, marker="s", lw=2, color=OM_COLOR, label="Omission ↓")
    handles = [l_ha, l_om]

    ax.set_xlabel(x_label)
    ax.set_ylabel("Hallucination ↓ / Omission ↓")

    # Values that must stay in view (curves, plus baselines if drawn).
    vals = ha + om
    if baseline_agg is not None and baseline_x is not None and baseline_x in baseline_agg:
        ha_b = baseline_agg[baseline_x]["ha_metric"][0]
        om_b = baseline_agg[baseline_x]["om_metric"][0]
        b_ha = ax.axhline(ha_b, color=HA_COLOR, linestyle="--", lw=1.6,
                          label=f"Hallucination ({baseline_ref_label})")
        b_om = ax.axhline(om_b, color=OM_COLOR, linestyle="--", lw=1.6,
                          label=f"Omission ({baseline_ref_label})")
        handles += [b_ha, b_om]
        vals += [ha_b, om_b]

    # Pad the combined range: a small margin below so nothing hugs the x-axis,
    # and more headroom above so the (now up to 4-entry) legend clears the curves.
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    top_pad = 0.42 if len(handles) > 2 else 0.28
    ax.set_ylim(lo - 0.10 * span, hi + top_pad * span)

    ax.grid(alpha=0.3)
    ax.legend(handles=handles, loc="upper center", ncol=2,
              frameon=True, framealpha=0.9, fontsize=8)


def save_one_curve(agg_mode, gs_list, out_path, title, x_label,
                   baseline_x=None, panel_metrics=None,
                   baseline_agg=None, baseline_label="Ingen guidance (baseline)",
                   baseline_ref_label="CFG, gs=1"):
    """Panel plot for ONE method (CFG or CFG++) — single curve per panel vs gs.

    Layout adapts to the number of panels: 2 → 1×2 row, 4 → 2×2 grid.
    If `baseline_x` is given AND present in the baseline source, draws a horizontal
    red dashed line at each panel's metric value at that x — the no-guidance reference.

    `baseline_agg` selects WHERE the baseline value is read from. It defaults to
    `agg_mode` (the same method being plotted). For the CFG++ plot we pass the
    standard-CFG aggregates instead, so the dashed line is plain conditional
    generation (CFG gs=1) — CFG++ has no neutral point of its own (every λ already
    renoises with e_uncond).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if panel_metrics is None:
        panel_metrics = PANEL_METRICS_FULL
    if baseline_agg is None:
        baseline_agg = agg_mode
    n = len(panel_metrics)
    if n <= 2:
        nrows, ncols, figsize = 1, n, (5.5 * n, 4.5)
    else:
        nrows, ncols, figsize = 2, 2, (11, 7.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = list(axes.flat)

    baseline_present = baseline_x is not None and baseline_x in baseline_agg
    for ax, (key, label) in zip(axes_flat, panel_metrics):
        if key == OM_HA_PANEL:
            _draw_om_ha_panel(ax, agg_mode, gs_list, x_label,
                              baseline_agg=baseline_agg, baseline_x=baseline_x,
                              baseline_ref_label=baseline_ref_label)
            continue
        ys = [agg_mode[g][key][0] for g in gs_list]
        ax.plot(gs_list, ys, marker="o", lw=2, label="Gennemsnit af testbilleder")
        if baseline_present:
            baseline_y = baseline_agg[baseline_x][key][0]
            ax.axhline(baseline_y, color="tab:red", linestyle="--", lw=1.8,
                       label=baseline_label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", frameon=True)

        # Add // break marks if the y-axis does not start at 0
        if ax.get_ylim()[0] > 0:
            d = 0.015
            kwargs = dict(transform=ax.transAxes, color="black", clip_on=False, lw=1.5)
            ax.plot((-d, d), (-d, d), **kwargs)
            ax.plot((-d, d), (-d + 0.02, d + 0.02), **kwargs)
    # Hide any unused subplot cells (e.g. odd panel count with 2×2 layout).
    for ax in axes_flat[len(panel_metrics):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_final_outputs(args, rows, run_dir, sweeps, stage, cmd_line):
    agg = aggregate(rows, sweeps)
    n_tiles = len({r["tile"] for r in rows})

    # Which metrics get plotted/printed depends on whether opensr_test was used.
    panel_metrics = PANEL_METRICS_PIXEL if getattr(args, "no_opensr_test", False) else PANEL_METRICS_FULL

    # Resolve the effective eta (falls back to the config value when --eta was omitted)
    # so the CSV records what actually ran, not just the literal command line.
    eff_eta = args.eta
    if eff_eta is None:
        eff_eta = float(OmegaConf.load(args.config).denoiser_settings.sampling_eta)
    seed_str = args.seed if (args.seed is not None and args.seed >= 0) else "disabled"

    out_csv = run_dir / "metrics.csv"
    fields = ["tile", "gs", "cfg_pp", *METRIC_COLS]
    with open(out_csv, "w", newline="") as f:
        f.write(f"# {cmd_line}\n")
        f.write(f"# effective: seed={seed_str} eta={eff_eta} "
                f"sampling_steps={args.sampling_steps}\n")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        for cfg_pp, gs_list in sweeps:
            cp = int(cfg_pp)
            tag = "cfgpp" if cp == 1 else "cfg"
            for gs in gs_list:
                row = {"tile": f"mean_{tag}_gs={gs}", "gs": gs, "cfg_pp": cp}
                row.update({k: agg[cp][gs][k][0] for k in METRIC_COLS})
                writer.writerow(row)
                row = {"tile": f"std_{tag}_gs={gs}", "gs": gs, "cfg_pp": cp}
                row.update({k: agg[cp][gs][k][1] for k in METRIC_COLS})
                writer.writerow(row)

    # Expand the twin sentinel into its two real metric columns for the table.
    table_keys = []
    for k, _ in panel_metrics:
        table_keys.extend(["ha_metric", "om_metric"] if k == OM_HA_PANEL else [k])

    print(f"\n  Sweep over {n_tiles} tiles  (sampling_steps={args.sampling_steps})")
    print(f"  {'mode':>6}  {'gs':>5}  " +
          "  ".join(f"{k:>10}" for k in table_keys))
    for cfg_pp, gs_list in sweeps:
        cp = int(cfg_pp)
        for gs in gs_list:
            row = "  ".join(f"{agg[cp][gs][k][0]:10.4f}" for k in table_keys)
            mode = "CFG++" if cp == 1 else "CFG"
            print(f"  {mode:>6}  {gs:5.2f}  {row}")
    print(f"\nPer-(tile, gs, mode) results saved to {out_csv}")

    if args.save_plot:
        for cfg_pp, gs_list in sweeps:
            cp = int(cfg_pp)
            tag = "cfgpp" if cp == 1 else "cfg"
            method = "CFG++" if cp == 1 else "CFG"
            x_label = "λ (CFG++)" if cp == 1 else "Guidance scale"
            # The neutral "no-guidance" reference is plain conditional DDIM =
            # standard CFG at gs=1.0. Draw that on BOTH plots: for CFG it lives in
            # its own aggregates; for CFG++ we cross-plot the CFG aggregates, since
            # no CFG++ λ equals plain conditional (every λ renoises with e_uncond).
            baseline_x = 1.0
            if cp == 1:
                baseline_agg = agg.get(0)               # standard-CFG aggregates
                baseline_label = "Ingen guidance (CFG, gs=1)"
            else:
                baseline_agg = None                      # use CFG's own gs=1.0
                baseline_label = "Ingen guidance (baseline)"
            plot_path = run_dir / f"curve_{tag}.png"
            title = (f"Effekt af {method} på {stage} UNet"
                     if stage else f"Effekt af {method}")
            save_one_curve(agg[cp], gs_list, plot_path, title, x_label,
                           baseline_x=baseline_x, panel_metrics=panel_metrics,
                           baseline_agg=baseline_agg, baseline_label=baseline_label)
            print(f"Plot saved to {plot_path}")


def write_shard_csv(rows, run_dir, shard_idx):
    out = run_dir / f"metrics.shard{shard_idx}.csv"
    fields = ["tile", "gs", "cfg_pp", *METRIC_COLS]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[shard {shard_idx}] wrote {out} ({len(rows)} rows)")


def read_shard_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {"tile": r["tile"], "gs": float(r["gs"]), "cfg_pp": int(r["cfg_pp"])}
            for k in METRIC_COLS:
                row[k] = float(r[k])
            rows.append(row)
    return rows


def run_multi_gpu(args, sweeps, n_devices):
    run_dir, stage = resolve_run_dir(args)
    cmd_line = "python " + shlex.join(sys.argv)
    script = str(pathlib.Path(__file__).resolve())

    print(f"Multi-GPU mode: launching {n_devices} shard processes")
    procs = []
    for i in range(n_devices):
        cmd = [sys.executable, script,
               "--unet_ckpt", args.unet_ckpt,
               "--npz_dir", args.npz_dir,
               "--config", args.config,
               "--pad_size", str(args.pad_size),
               "--gs", args.gs,
               "--sampling_steps", str(args.sampling_steps),
               "--batch_size", str(args.batch_size),
               "--out_dir", args.out_dir,
               "--shard", f"{i}/{n_devices}",
               "--devices", "1",
               "--device", "cuda:0"]
        if args.max_tiles is not None:
            cmd += ["--max_tiles", str(args.max_tiles)]
        if args.eta is not None:
            cmd += ["--eta", str(args.eta)]
        cmd += ["--seed", str(args.seed)]
        if args.cfgpp:
            cmd += ["--cfgpp", "--gs_cfgpp", args.gs_cfgpp]
        if args.no_opensr_test:
            cmd += ["--no_opensr_test"]
        if args.save_images:
            cmd += ["--save_images"]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        procs.append((i, subprocess.Popen(cmd, env=env)))

    failures = [i for i, p in procs if p.wait() != 0]
    if failures:
        print(f"ERROR: shard(s) {failures} failed.")
        sys.exit(1)

    all_rows = []
    for i in range(n_devices):
        shard_csv = run_dir / f"metrics.shard{i}.csv"
        all_rows.extend(read_shard_csv(shard_csv))
        shard_csv.unlink()
    write_final_outputs(args, all_rows, run_dir, sweeps, stage, cmd_line)


def main():
    parser = argparse.ArgumentParser(description="CFG / CFG++ impact diagnostic")
    parser.add_argument("--unet_ckpt", required=True)
    parser.add_argument("--npz_dir", required=True)
    parser.add_argument("--config",
                        default=str(ROOT / "opensr_model" / "configs" / "config_1m.yaml"),
                        help="1m: config_1m.yaml; 5m: config_10m.yaml")
    parser.add_argument("--pad_size", type=int, required=True,
                        help="VAE training size (1024 for 1m, 256 for 5m). "
                             "Currently informational — model.forward handles padding internally.")
    parser.add_argument("--gs", default="1.0,2.0,3.0,4.0,5.0,6.0,7.5,10.0",
                        help="Comma-separated guidance_scale values for the standard CFG sweep")
    parser.add_argument("--cfgpp", action="store_true",
                        help="Also run a CFG++ sweep at small λ values (and save a second plot). "
                             "CFG++ uses a fundamentally different scale range than standard CFG "
                             "(λ ∈ (0, 1] per Chung et al. 2024).")
    parser.add_argument("--gs_cfgpp", default="0.1,0.3,0.5,0.6,0.7,0.9,1.0",
                        help="Comma-separated λ values for the CFG++ sweep (only used with --cfgpp). "
                             "λ ∈ (0,1]: λ=1 is fully conditional, λ≈0.6 is the paper's sweet spot. "
                             "Do NOT include λ=0 — that is purely unconditional sampling (ignores "
                             "the S1/S2 conditioning and hallucinates an unrelated scene).")
    parser.add_argument("--sampling_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed applied before every forward so CFG and CFG++ share the "
                             "same initial latent (controlled comparison). Set to a different "
                             "int to vary, or pass a negative value to disable seeding.")
    parser.add_argument("--eta", type=float, default=None,
                        help="DDIM eta: 0=deterministic, 1=fully stochastic (DDPM). "
                             "Default: the config value (0.95). Use --eta 0 to match the "
                             "CFG++ paper's deterministic DDIM and remove sampling noise.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Tiles per batched forward (raise for better GPU usage; "
                             "lower if OOM). 5m can usually take 8+; 1m around 4.")
    parser.add_argument("--out_dir", default=str(ROOT / "test" / "results" / "cfg-sweep"))
    parser.add_argument("--max_tiles", type=int, default=None,
                        help="Eval at most this many tiles (recommended 20-30 for tractable wall-time).")
    parser.add_argument("--save_plot", action="store_true")
    parser.add_argument("--no_opensr_test", action="store_true",
                        help="Skip the opensr_test consistency/correctness metrics "
                             "(use for the 1m stage, where the 10× scale breaks satalign). "
                             "Plot then shows only PSNR + SSIM panels.")
    parser.add_argument("--save_images", action="store_true",
                        help="Save per-tile, per-mode comparison grids "
                             "(Original + SR@each gs in a single row) under <run_dir>/images/. "
                             "One PNG per (tile, mode). Adds a bit of disk; no metric cost.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--devices", type=int,
                        default=(torch.cuda.device_count() if torch.cuda.is_available() else 1))
    parser.add_argument("--shard", type=str, default=None)
    args = parser.parse_args()

    # Build the sweep list. Always include standard CFG; optionally add CFG++ at the small range.
    gs_values = sorted(float(s) for s in args.gs.split(","))
    assert all(g > 0 for g in gs_values), "--gs values must be > 0"
    sweeps = [(False, gs_values)]
    if args.cfgpp:
        gs_cfgpp_values = sorted(float(s) for s in args.gs_cfgpp.split(","))
        assert all(0 <= g <= 1 for g in gs_cfgpp_values), \
            "--gs_cfgpp values should be in [0, 1] (CFG++ uses a small-λ scale)"
        sweeps.append((True, gs_cfgpp_values))

    if args.shard is not None:
        shard_idx, shard_count = (int(x) for x in args.shard.split("/"))
        rows = run_eval_loop(args, sweeps, shard_idx, shard_count)
        run_dir, _ = resolve_run_dir(args)
        write_shard_csv(rows, run_dir, shard_idx)
        return

    if args.devices > 1:
        run_multi_gpu(args, sweeps, args.devices)
        return

    rows = run_eval_loop(args, sweeps)
    run_dir, stage = resolve_run_dir(args)
    cmd_line = "python " + shlex.join(sys.argv)
    write_final_outputs(args, rows, run_dir, sweeps, stage, cmd_line)


if __name__ == "__main__":
    main()
