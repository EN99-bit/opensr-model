"""Inference-speed benchmark across all models used in the thesis.

Times the actual super-resolution forward/sampling per tile (warm-up + N timed
runs, CUDA-synchronised), excluding model loading and disk I/O.

Produces:
  A. Per-model timing at gs=1 and gs=2 (100 steps).
  B. 5m S1+S2 timing across sampling steps (gs=1).
"""
import sys, time, pathlib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from opensr_model.srmodel import SRLatentDiffusion
from opensr_model.data import FusionDataset
from opensr_model.utils import normalize_s2, normalize_s1, normalize_aerial

DEV = "cuda"
WARMUP, TIMED = 1, 3
AERIAL_5M_PAD, AERIAL_1M_PAD, LATENT_1M = 256, 1024, 128
C = ROOT / "opensr_model" / "configs"

def load_weights(model, ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.model.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("ldm.")}, strict=False)

def s2_only_encode(model):
    def enc(X_s2, X_s1):
        model._X_s2 = X_s2.clone()
        hr = X_s2.shape[-1] * model.scale_factor
        up = F.interpolate(normalize_s2(X_s2, stage="norm"), size=(hr, hr), mode="bilinear", align_corners=False)
        return model.model.first_stage_model.encode(up).mode().to(model.device)
    return enc

@torch.no_grad()
def run_stage2(m2, sr_5m, s1_128, s2_128, steps, gs):
    dt = next(m2.model.first_stage_model.parameters()).dtype
    if sr_5m.shape[-1] != AERIAL_5M_PAD:
        sr_5m = F.interpolate(sr_5m, (AERIAL_5M_PAD, AERIAL_5M_PAD), mode="bilinear", align_corners=False)
    a_up = F.interpolate(normalize_aerial(sr_5m.to(DEV), stage="norm").to(dt), (AERIAL_1M_PAD, AERIAL_1M_PAD), mode="bilinear", align_corners=False)
    z_5m = m2.model.first_stage_model.encode(a_up).mode()
    s1_up = F.interpolate(normalize_s1(s1_128.to(DEV), stage="norm").to(dt), (LATENT_1M, LATENT_1M), mode="bilinear", align_corners=False)
    s2_up = F.interpolate(normalize_s2(s2_128.to(DEV), stage="norm").to(dt), (AERIAL_1M_PAD, AERIAL_1M_PAD), mode="bilinear", align_corners=False)
    z_s2 = m2.model.first_stage_model.encode(s2_up).mode()
    cond = torch.cat([z_5m, s1_up, z_s2], dim=1); null = torch.zeros_like(cond)
    ddim, latent, time_range = m2._prepare_model(cond, custom_steps=steps)
    for i, step in enumerate(time_range):
        index = steps - i - 1; t = torch.full((1,), step, device=DEV, dtype=torch.long)
        if gs > 1.0:
            e_u = m2.model.apply_model(latent, t, cond=null); e_c = m2.model.apply_model(latent, t, cond=cond)
            latent = m2._ddim_step(latent, e_u + gs * (e_c - e_u), index, ddim, 1.0)
        else:
            latent, _ = ddim.p_sample_ddim(x=latent, c=cond, t=step, index=index, use_original_steps=False, temperature=1.0)
    return m2._tensor_decode(latent, spe_cor=False)

def timeit(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(TIMED):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return float(np.mean(ts))

def build(cfg_name, ckpt, opensr=False, no_s1=False):
    model = SRLatentDiffusion(OmegaConf.load(C / cfg_name), device=DEV)
    if opensr:
        model.load_pretrained(OmegaConf.load(C / cfg_name).ckpt_version); model.scale_factor = 4
        model._tensor_encode = s2_only_encode(model)
    else:
        load_weights(model, ckpt)
        if no_s1: model._tensor_encode = s2_only_encode(model)
    model.eval(); return model

def main():
    ds = FusionDataset(str(pathlib.Path("~/npz/apr2025/5m-untouched").expanduser()), require_aerial=False, pad=True)
    s = ds[0]; s2p, s1p = s["s2"].unsqueeze(0), s["s1"].unsqueeze(0)
    s2n, s1n = s2p[:, :, 14:114, 14:114], s1p[:, :, 14:114, 14:114]
    def F1(m, a2, a1): return lambda st, gs: m.forward(a2.to(DEV), a1.to(DEV), sampling_steps=st, guidance_scale=gs, histogram_matching=False)

    gs_tbl, step_tbl = [], []
    bic = timeit(lambda: F.interpolate(s2n.to(DEV), scale_factor=2, mode="bicubic", align_corners=False))
    gs_tbl.append(("Bikubisk", bic, bic)); print("bicubic done")

    specs = [
        ("5m S1+S2",       "config_10m.yaml",       "checkpoints/5m/unet-no-latents/unet-epoch=0098-val_loss=0.102384.ckpt", {}, s2p, s1p),
        ("5m S2 uden S1",  "config_10m_no_s1.yaml", "checkpoints/5m/unet-no-s1-matched/unet-no-s1-epoch=0079-val_loss=0.111461.ckpt", dict(no_s1=True), s2p, s1p),
        ("Direkte 10m-1m", "config_1m.yaml",        "checkpoints/1m/unet/unet-epoch=0891-val_loss=0.216656.ckpt", {}, s2n, s1n),
        ("LDSR-S2",        "config_opensr.yaml",    None, dict(opensr=True), s2n, s1n),
    ]
    for name, cfg, ckpt, kw, a2, a1 in specs:
        m = build(cfg, str(ROOT/ckpt) if ckpt else None, **kw); f = F1(m, a2, a1)
        gs_tbl.append((name, timeit(lambda: f(100, 1.0)), timeit(lambda: f(100, 2.0))))
        if name == "5m S1+S2":
            for st in [1, 10, 25, 50, 100]: step_tbl.append((st, timeit(lambda st=st: f(st, 1.0))))
        print(f"{name} done"); del m; torch.cuda.empty_cache()

    m1 = build("config_10m.yaml", str(ROOT/"checkpoints/5m/unet-no-latents/unet-epoch=0098-val_loss=0.102384.ckpt"))
    m2 = build("config_5m_to_1m_with_s2.yaml", str(ROOT/"checkpoints/5to1m_with_s2/unet/unet5to1-s1s2-epoch=0246-val_loss=0.176670.ckpt"))
    def casc(gs):
        sr5 = m1.forward(s2p.to(DEV), s1p.to(DEV), sampling_steps=100, guidance_scale=gs, histogram_matching=False)
        return run_stage2(m2, sr5, s1p, s2p, 100, gs)
    gs_tbl.append(("Kaskade 10m-1m", timeit(lambda: casc(1.0)), timeit(lambda: casc(2.0)))); print("cascade done")

    import csv
    out = ROOT / "test" / "results" / "timing"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "timing_gs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "s_per_tile_gs1", "s_per_tile_gs2"])
        for n, g1, g2 in gs_tbl: w.writerow([n, f"{g1:.4f}", f"{g2:.4f}"])
    with open(out / "timing_steps.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["steps", "s_per_tile_5m_s1s2_gs1"])
        for st, t in step_tbl: w.writerow([st, f"{t:.4f}"])

    print("\n=== A. s/tile at 100 steps ===")
    print(f"{'Model':<18}{'gs=1':>9}{'gs=2':>9}")
    for n, g1, g2 in gs_tbl: print(f"{n:<18}{g1:>9.3f}{g2:>9.3f}")
    print("\n=== B. 5m S1+S2, s/tile vs steps (gs=1) ===")
    for st, t in step_tbl: print(f"  steps={st:<4} {t:.3f}")
    print(f"\nSaved {out/'timing_gs.csv'} and {out/'timing_steps.csv'}")

if __name__ == "__main__":
    main()
