"""DDIM sampling-steps sweep: how few steps until quality converges.

Companion to test/eval_cfg.py. Holds guidance fixed and sweeps the number of
DDIM sampling steps, scoring each with the same opensr_test + PSNR/SSIM metrics.
It reuses eval_cfg's scoring (`score_batch`), plotting (`save_one_curve`, the
omission/hallucination panel) and per-tile image grids, so the two diagnostics
stay visually consistent.

x-axis: sampling steps. Reference line on every panel: the highest step count in
the sweep (treated as the converged result) — so you can read how far each
lower step count sits from convergence.

Usage:
    # 5m
    python test/eval_sample_steps.py \
        --unet_ckpt checkpoints/5m/unet-no-latents/unet-epoch=0098-val_loss=0.102384.ckpt \
        --config opensr_model/configs/config_10m.yaml \
        --npz_dir ~/npz/apr2025/5m-untouched --pad_size 256 \
        --steps 10,25,50,75,100 --eta 0 --save_plot --save_images

    # 1m (opensr_test still runs via the LR-upsample workaround; slower)
    python test/eval_sample_steps.py \
        --unet_ckpt checkpoints/1m/unet/unet-epoch=0891-val_loss=0.216656.ckpt \
        --config opensr_model/configs/config_1m.yaml \
        --npz_dir ~/npz/apr2025/1m-untouched --pad_size 1024 \
        --steps 10,25,50,75,100 --eta 0 --save_plot
"""

import argparse
import csv
import os
import pathlib
import shlex
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

# Reuse eval_cfg's scoring + plotting so both diagnostics stay consistent.
import eval_cfg
from eval_cfg import (
    score_batch, save_metric_plots, save_tile_grid, resolve_run_dir,
    resolve_cond_mode, batch_cond_5m, aerial_rgb_u8, AERIAL_5M_NATIVE,
    lpips_score, to_rgb_u8_norm, METRIC_COLS, OPENSR_METRICS,
    PANEL_METRICS_FULL, PANEL_METRICS_PIXEL,
)
from opensr_model.srmodel import SRLatentDiffusion
from opensr_model.data import FusionDataset
from opensr_model.utils import normalize_s2
from eval import load_trained_weights


def bicubic_baseline(samples, device, no_opensr=False):
    """'No-model' floor (plotted at x=0): bicubic-upsample the S2 input to the aerial
    resolution and score it against the GT aerial with the same metrics. Returns
    (per-tile metric dicts, RGB thumbs). Same LR-upsample workaround as score_batch.
    """
    results, thumbs = [], []
    for s in samples:
        s2 = s["s2"].to(device).float()
        aerial = s["aerial"].to(device).float()
        H = aerial.shape[-1]
        lr_norm = (normalize_s2(s2, stage="norm") + 1.0) / 2.0
        sr_norm = F.interpolate(lr_norm.unsqueeze(0), size=(H, H),
                                mode="bicubic", align_corners=False).clamp(0, 1)[0]
        hr_norm = (aerial / 255.0).clamp(0, 1)
        out = {}
        if no_opensr or getattr(eval_cfg, "opensr_test", None) is None:
            for k in OPENSR_METRICS:
                out[k] = float("nan")
        else:
            try:
                lr_eval = lr_norm
                if hr_norm.shape[-1] / lr_norm.shape[-1] > 4:
                    t = hr_norm.shape[-1] // 4
                    lr_eval = F.interpolate(lr_norm.unsqueeze(0), size=(t, t),
                                            mode="bilinear", align_corners=False)[0]
                r = eval_cfg.opensr_test.Metrics().compute(lr=lr_eval, sr=sr_norm, hr=hr_norm)
                for k in OPENSR_METRICS:
                    out[k] = float(r.get(k, float("nan")))
            except Exception:
                for k in OPENSR_METRICS:
                    out[k] = float("nan")
        hr_u8 = to_rgb_u8_norm(hr_norm)
        sr_u8 = to_rgb_u8_norm(sr_norm)
        out["psnr"] = float(peak_signal_noise_ratio(hr_u8, sr_u8, data_range=255))
        out["ssim"] = float(structural_similarity(hr_u8, sr_u8, channel_axis=2, data_range=255))
        out["lpips"] = lpips_score(sr_norm, hr_norm, device)
        results.append(out)
        thumbs.append(sr_u8)
    return results, thumbs


def aggregate_steps(rows, steps_list):
    """{steps: {metric: (mean, std)}} — mean/std across tiles at each step count."""
    out = {s: {} for s in steps_list}
    for s in steps_list:
        sel = [r for r in rows if r["steps"] == s]
        for k in METRIC_COLS:
            v = np.array([r[k] for r in sel], dtype=np.float64)
            v = v[~np.isnan(v)]
            out[s][k] = (float(v.mean()), float(v.std())) if len(v) else (float("nan"), float("nan"))
    return out


def run_eval_loop(args, steps_list, shard_idx=0, shard_count=1):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"[shard {shard_idx}/{shard_count}]" if shard_count > 1 else ""
    print(f"{tag} Device: {device}")

    cfg = OmegaConf.load(args.config)
    model = SRLatentDiffusion(cfg, device=device)
    print(f"{tag} Loading UNet (+ bundled VAE): {args.unet_ckpt}")
    load_trained_weights(model, args.unet_ckpt)
    model = model.to(device).eval()

    cfg_pp = bool(args.cfgpp)
    mode_label = "CFG++" if cfg_pp else "CFG"
    print(f"{tag} steps sweep={steps_list}  guidance={args.gs} ({mode_label})  eta={args.eta}")

    cond_mode, variant, stage1, gt_map = resolve_cond_mode(args, cfg, device, tag)

    ds = FusionDataset(args.npz_dir, require_aerial=True, pad=False)
    if len(ds) == 0:
        print("No valid tiles found.")
        return []
    n_total = len(ds) if args.max_tiles is None else min(args.max_tiles, len(ds))
    tile_indices = list(range(shard_idx, n_total, shard_count))

    bs = max(1, args.batch_size)
    tile_batches = [tile_indices[i:i + bs] for i in range(0, len(tile_indices), bs)]
    print(f"{tag} Evaluating {len(tile_indices)} tiles in {len(tile_batches)} batches "
          f"of up to {bs} × {len(steps_list)} runs each")

    save_images = args.save_images
    if save_images:
        images_dir, _ = resolve_run_dir(args)
        images_dir = images_dir / "images"
        images_dir.mkdir(exist_ok=True)

    show_progress = (shard_idx == 0)
    rows = []
    for batch_indices in tqdm(tile_batches, desc=f"Steps sweep {tag}".strip(),
                              disable=not show_progress):
        samples = [ds[i] for i in batch_indices]
        names = [pathlib.Path(s["path"]).stem for s in samples]

        # Stage-2 conditioning (None = direct S2+S1), built once per batch.
        cond_5m, samples, names = batch_cond_5m(
            cond_mode, samples, names, device, stage1, gt_map, getattr(args, "stage1_steps", 100))
        if not samples:
            continue
        cond5m_rgb = {n: aerial_rgb_u8(cond_5m[i], AERIAL_5M_NATIVE) for i, n in enumerate(names)} if cond_5m is not None else {}

        tile_srs = {} if save_images else None

        for steps in steps_list:
            args.sampling_steps = steps   # score_batch reads args.sampling_steps
            batch_results, batch_thumbs = score_batch(model, samples, args, args.gs, cfg_pp,
                                                       cond_5m=cond_5m, variant=variant)
            for name, m in zip(names, batch_results):
                rows.append({"tile": name, "steps": steps,
                             **{k: m[k] for k in METRIC_COLS}})
            if tile_srs is not None:
                for name, thumb in zip(names, batch_thumbs):
                    tile_srs.setdefault(name, {})[steps] = thumb

        # Bicubic 'no-model' floor: plotted at x=0 (steps=0) and shown as a grid panel.
        b_results, b_thumbs = bicubic_baseline(samples, device, no_opensr=args.no_opensr_test)
        b_thumb_by_name = {}
        for name, m, th in zip(names, b_results, b_thumbs):
            rows.append({"tile": name, "steps": 0, **{k: m[k] for k in METRIC_COLS}})
            b_thumb_by_name[name] = th

        if tile_srs is not None:
            for sample in samples:
                name = pathlib.Path(sample["path"]).stem
                hr_rgb = (sample["aerial"][:3].numpy().transpose(1, 2, 0)
                          .clip(0, 255).astype(np.uint8))
                if name in tile_srs:
                    out_path = images_dir / f"{name}_steps.png"
                    extra = []
                    if name in cond5m_rgb:
                        lbl = "5m (predicted)" if cond_mode == "cascade" else "5m (GT)"
                        extra.append((lbl, cond5m_rgb[name]))
                    extra.append(("Bicubic", b_thumb_by_name[name]))
                    save_tile_grid(name, hr_rgb, tile_srs[name], mode_label, out_path,
                                   label_fmt="{:g} steps", extra_panels=extra)
    return rows


def write_shard_csv(rows, run_dir, shard_idx):
    out = run_dir / f"metrics_steps.shard{shard_idx}.csv"
    fields = ["tile", "steps", *METRIC_COLS]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[shard {shard_idx}] wrote {out} ({len(rows)} rows)")


def read_shard_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {"tile": r["tile"], "steps": int(r["steps"])}
            for k in METRIC_COLS:
                row[k] = float(r[k])
            rows.append(row)
    return rows


def run_multi_gpu(args, steps_list, n_devices):
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
               "--steps", args.steps,
               "--gs", str(args.gs),
               "--batch_size", str(args.batch_size),
               "--out_dir", args.out_dir,
               "--seed", str(args.seed),
               "--shard", f"{i}/{n_devices}",
               "--devices", "1",
               "--device", "cuda:0"]
        if args.max_tiles is not None:
            cmd += ["--max_tiles", str(args.max_tiles)]
        if args.eta is not None:
            cmd += ["--eta", str(args.eta)]
        if args.cfgpp:
            cmd += ["--cfgpp"]
        if args.no_opensr_test:
            cmd += ["--no_opensr_test"]
        if args.save_images:
            cmd += ["--save_images"]
        if getattr(args, "cond_5m_dir", None):
            cmd += ["--cond_5m_dir", args.cond_5m_dir]
        if getattr(args, "cascade", False):
            cmd += ["--cascade", "--stage1_ckpt", args.stage1_ckpt,
                    "--stage1_config", args.stage1_config,
                    "--stage1_steps", str(args.stage1_steps)]
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
        shard_csv = run_dir / f"metrics_steps.shard{i}.csv"
        all_rows.extend(read_shard_csv(shard_csv))
        shard_csv.unlink()
    write_outputs(args, all_rows, run_dir, steps_list, stage, cmd_line)


def write_outputs(args, rows, run_dir, steps_list, stage, cmd_line):
    # Derive the axis from the rows so the bicubic floor (steps=0) is included.
    steps_list = sorted({int(r["steps"]) for r in rows})
    agg = aggregate_steps(rows, steps_list)
    n_tiles = len({r["tile"] for r in rows})
    panel_metrics = PANEL_METRICS_PIXEL if args.no_opensr_test else PANEL_METRICS_FULL

    eff_eta = args.eta
    if eff_eta is None:
        eff_eta = float(OmegaConf.load(args.config).denoiser_settings.sampling_eta)
    seed_str = args.seed if (args.seed is not None and args.seed >= 0) else "disabled"
    mode_label = "CFG++" if args.cfgpp else "CFG"

    out_csv = run_dir / "metrics_steps.csv"
    fields = ["tile", "steps", *METRIC_COLS]
    with open(out_csv, "w", newline="") as f:
        f.write(f"# {cmd_line}\n")
        f.write(f"# effective: seed={seed_str} eta={eff_eta} guidance={args.gs} "
                f"mode={mode_label}\n")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        for s in steps_list:
            writer.writerow({"tile": f"mean_steps={s}", "steps": s,
                             **{k: agg[s][k][0] for k in METRIC_COLS}})
            writer.writerow({"tile": f"std_steps={s}", "steps": s,
                             **{k: agg[s][k][1] for k in METRIC_COLS}})

    # Console table
    table_keys = []
    for k, _ in panel_metrics:
        table_keys.extend(["ha_metric", "om_metric"] if k == eval_cfg.OM_HA_PANEL else [k])
    print(f"\n  Steps sweep over {n_tiles} tiles  (guidance={args.gs} {mode_label}, eta={eff_eta})")
    print(f"  {'steps':>6}  " + "  ".join(f"{k:>10}" for k in table_keys))
    for s in steps_list:
        row = "  ".join(f"{agg[s][k][0]:10.4f}" for k in table_keys)
        print(f"  {s:6d}  {row}")
    print(f"\nPer-(tile, steps) results saved to {out_csv}")

    if args.save_plot:
        # One plot file per metric in steps_plots/. The dashed reference is the bicubic
        # 'no-model' floor (steps=0); the curve covers the actual step counts (>0).
        curve_steps = [s for s in steps_list if s > 0]
        metrics_to_plot = (["psnr", "ssim", "lpips"]
                           if args.no_opensr_test else METRIC_COLS)
        plots_dir = run_dir / "steps_plots"
        save_metric_plots(agg, curve_steps, plots_dir, "Sampling steps", metrics_to_plot,
                          baseline_x=0, baseline_label="Bicubic (ingen model)")
        print(f"Per-metric plots saved to {plots_dir}/ ({len(metrics_to_plot)} metrics)")


def main():
    parser = argparse.ArgumentParser(
        description="DDIM sampling-steps sweep (companion to eval_cfg.py)")
    parser.add_argument("--unet_ckpt", required=True)
    parser.add_argument("--npz_dir", required=True)
    parser.add_argument("--config",
                        default=str(ROOT / "opensr_model" / "configs" / "config_1m.yaml"),
                        help="1m: config_1m.yaml; 5m: config_10m.yaml")
    parser.add_argument("--pad_size", type=int, required=True,
                        help="VAE training size (1024 for 1m, 256 for 5m). "
                             "Informational — model.forward handles padding internally.")
    parser.add_argument("--steps", default="1,2,5,10,20,25,50,100",
                        help="Comma-separated DDIM step counts to sweep. Prefer counts that "
                             "divide 1000 evenly (1,5,10,20,25,50,100,...) — others (e.g. 75) hit "
                             "a schedule-length/index mismatch in forward and diverge.")
    parser.add_argument("--gs", type=float, default=1.0,
                        help="Fixed guidance held constant across the sweep. With --cfgpp "
                             "this is λ∈(0,1]; otherwise it is the CFG scale (≥1).")
    parser.add_argument("--cfgpp", action="store_true",
                        help="Hold guidance fixed in CFG++ mode (λ) instead of standard CFG.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed applied before every forward so each step count starts "
                             "from the same initial latent. Negative disables.")
    parser.add_argument("--eta", type=float, default=None,
                        help="DDIM eta: 0=deterministic, 1=stochastic. Default: config value. "
                             "Use --eta 0 for a clean, reproducible steps comparison.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_dir", default=str(ROOT / "test" / "results" / "steps-sweep"))
    parser.add_argument("--max_tiles", type=int, default=None)
    parser.add_argument("--save_plot", action="store_true")
    parser.add_argument("--save_images", action="store_true",
                        help="Save per-tile grids (Original + SR@each step count).")
    parser.add_argument("--no_opensr_test", action="store_true",
                        help="Skip opensr_test metrics (plot shows only PSNR + SSIM).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--devices", type=int,
                        default=(torch.cuda.device_count() if torch.cuda.is_available() else 1),
                        help="GPUs to shard tiles across (>1 launches subprocesses).")
    parser.add_argument("--shard", type=str, default=None,
                        help="Internal: 'idx/count' for a single shard worker.")
    # Stage-2 conditioning modes (5→1m cascade models). Default: direct S2+S1.
    parser.add_argument("--cond_5m_dir", type=str, default=None,
                        help="Stage-2 ISOLATED: condition on GT 5m aerial from this dir.")
    parser.add_argument("--cascade", action="store_true",
                        help="Full cascade: predict 5m via a stage-1 model, then condition "
                             "stage-2 on it. Needs --stage1_ckpt/_config.")
    parser.add_argument("--stage1_ckpt", type=str, default=None)
    parser.add_argument("--stage1_config", type=str, default=None)
    parser.add_argument("--stage1_steps", type=int, default=100)
    args = parser.parse_args()
    # score_batch reads this attribute; set a placeholder so the namespace is complete.
    args.sampling_steps = None

    assert not (args.cascade and args.cond_5m_dir), \
        "--cascade and --cond_5m_dir are mutually exclusive"
    if args.cascade:
        assert args.stage1_ckpt and args.stage1_config, \
            "--cascade requires --stage1_ckpt and --stage1_config"

    steps_list = sorted(int(s) for s in args.steps.split(","))
    assert all(s > 0 for s in steps_list), "--steps values must be > 0"
    if args.cfgpp:
        assert 0 < args.gs <= 1, "--gs is λ∈(0,1] when --cfgpp is set"

    if args.shard is not None:
        shard_idx, shard_count = (int(x) for x in args.shard.split("/"))
        rows = run_eval_loop(args, steps_list, shard_idx, shard_count)
        run_dir, _ = resolve_run_dir(args)
        write_shard_csv(rows, run_dir, shard_idx)
        return

    if args.devices > 1:
        run_multi_gpu(args, steps_list, args.devices)
        return

    run_dir, stage = resolve_run_dir(args)
    rows = run_eval_loop(args, steps_list)
    if not rows:
        return
    cmd_line = "python " + shlex.join(sys.argv)
    write_outputs(args, rows, run_dir, steps_list, stage, cmd_line)


if __name__ == "__main__":
    main()
