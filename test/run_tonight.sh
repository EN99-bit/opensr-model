#!/usr/bin/env bash
# Eval sweep: cfg + sample_steps + unet-oracle.
#   direct   : 5m, 5m-backup, 1m            (S2+S1 conditioning)
#   isolated : <stage-2> + GT 5m            (--cond_5m_dir)
#   cascade  : 5m → <stage-2>               (--cascade, predicted 5m)
#
# ACTIVE: the s1s2 stage-2 variant (isolated + cascade).
# The direct/s1 block below is COMMENTED OUT (already run) — uncomment to re-run.
#
# Run from the repo root with your usual python env active, e.g. in screen:
#   screen -S tonight     # then: bash test/run_tonight.sh 2>&1 | tee test/results/_logs/tonight.log
#
# Each run logs to test/results/_logs/<name>.log and FAILURES DO NOT STOP the rest.
set -u
cd "$(dirname "$0")/.."                      # repo root

LOGDIR=test/results/_logs
mkdir -p "$LOGDIR"

# ── checkpoints / configs / data ────────────────────────────────────────────
M5=checkpoints/5m/unet-no-latents/unet-epoch=0098-val_loss=0.102384.ckpt   # stage-1 / 5m direct
M5B=checkpoints/5m/backup-unet-epoch=0007-val_loss=0.300377.ckpt
M1=checkpoints/1m/unet/unet-epoch=0891-val_loss=0.216656.ckpt
M21=checkpoints/5to1m-s1/unet/unet5to1-epoch=0241-val_loss=0.176718.ckpt          # stage-2 (s1)
S2M=checkpoints/5to1m_with_s2/unet/unet5to1-s1s2-epoch=0246-val_loss=0.176670.ckpt # stage-2 (s1s2)
C10=opensr_model/configs/config_10m.yaml
C1=opensr_model/configs/config_1m.yaml
C12=opensr_model/configs/config_5m_to_1m_with_s2.yaml
N5="$HOME/npz/apr2025/5m-untouched"
N1="$HOME/npz/apr2025/1m-untouched"

run () {  # run <logname> <command...>
  local name="$1"; shift
  echo "===== [$(date '+%F %T')] START $name ====="
  if "$@" > "$LOGDIR/$name.log" 2>&1; then
    echo "===== [$(date '+%F %T')] DONE  $name ====="
  else
    echo "===== [$(date '+%F %T')] FAIL  $name  (rc=$?, see $LOGDIR/$name.log) ====="
  fi
}

ISO=(--cond_5m_dir "$N5")
CAS=(--cascade --stage1_ckpt "$M5" --stage1_config "$C10")

# ============================================================================
# ALREADY RUN (kept for reference / re-run — uncomment to use)
# ============================================================================
# # unet-oracle
# run unet_5m            python test/eval_unet.py        --unet_ckpt "$M5"  --config "$C10" --npz_dir "$N5" --pad_size 256  --save_plot --save_images
# run unet_5m_backup     python test/eval_unet.py        --unet_ckpt "$M5B" --config "$C10" --npz_dir "$N5" --pad_size 256  --save_plot --save_images
# run unet_1m            python test/eval_unet.py        --unet_ckpt "$M1"  --config "$C1"  --npz_dir "$N1" --pad_size 1024 --save_plot --save_images
# run unet_5to1m_iso     python test/eval_unet.py        --unet_ckpt "$M21" --config "$C1"  --npz_dir "$N1" --pad_size 1024 "${ISO[@]}" --save_plot --save_images
# run unet_5to1m_cascade python test/eval_unet.py        --unet_ckpt "$M21" --config "$C1"  --npz_dir "$N1" --pad_size 1024 "${CAS[@]}" --save_plot --save_images
# # cfg / cfg++
# run cfg_5m             python test/eval_cfg.py         --unet_ckpt "$M5"  --config "$C10" --npz_dir "$N5" --pad_size 256  --save_plot --cfgpp --save_images
# run cfg_5m_backup      python test/eval_cfg.py         --unet_ckpt "$M5B" --config "$C10" --npz_dir "$N5" --pad_size 256  --save_plot --cfgpp --save_images
# run cfg_1m             python test/eval_cfg.py         --unet_ckpt "$M1"  --config "$C1"  --npz_dir "$N1" --pad_size 1024 --save_plot --cfgpp --save_images
# run cfg_5to1m_iso      python test/eval_cfg.py         --unet_ckpt "$M21" --config "$C1"  --npz_dir "$N1" --pad_size 1024 "${ISO[@]}" --save_plot --cfgpp --save_images
# run cfg_5to1m_cascade  python test/eval_cfg.py         --unet_ckpt "$M21" --config "$C1"  --npz_dir "$N1" --pad_size 1024 "${CAS[@]}" --save_plot --cfgpp --save_images
# # sampling-steps
# run steps_5m            python test/eval_sample_steps.py --unet_ckpt "$M5"  --config "$C10" --npz_dir "$N5" --pad_size 256  --save_plot --save_images
# run steps_5m_backup     python test/eval_sample_steps.py --unet_ckpt "$M5B" --config "$C10" --npz_dir "$N5" --pad_size 256  --save_plot --save_images
# run steps_1m            python test/eval_sample_steps.py --unet_ckpt "$M1"  --config "$C1"  --npz_dir "$N1" --pad_size 1024 --save_plot --save_images
# run steps_5to1m_iso     python test/eval_sample_steps.py --unet_ckpt "$M21" --config "$C1"  --npz_dir "$N1" --pad_size 1024 "${ISO[@]}" --save_plot --save_images
# run steps_5to1m_cascade python test/eval_sample_steps.py --unet_ckpt "$M21" --config "$C1"  --npz_dir "$N1" --pad_size 1024 "${CAS[@]}" --save_plot --save_images

# ============================================================================
# TO RUN NOW — s1s2 stage-2 variant (isolated + cascade)
# ============================================================================
# unet-oracle (cheapest: single-step)
run unet_s1s2_iso      python test/eval_unet.py        --unet_ckpt "$S2M" --config "$C12" --npz_dir "$N1" --pad_size 1024 "${ISO[@]}" --save_plot --save_images
run unet_s1s2_cascade  python test/eval_unet.py        --unet_ckpt "$S2M" --config "$C12" --npz_dir "$N1" --pad_size 1024 "${CAS[@]}" --save_plot --save_images

# cfg / cfg++
run cfg_s1s2_iso       python test/eval_cfg.py         --unet_ckpt "$S2M" --config "$C12" --npz_dir "$N1" --pad_size 1024 "${ISO[@]}" --save_plot --cfgpp --save_images
run cfg_s1s2_cascade   python test/eval_cfg.py         --unet_ckpt "$S2M" --config "$C12" --npz_dir "$N1" --pad_size 1024 "${CAS[@]}" --save_plot --cfgpp --save_images

# sampling-steps (heaviest)
run steps_s1s2_iso     python test/eval_sample_steps.py --unet_ckpt "$S2M" --config "$C12" --npz_dir "$N1" --pad_size 1024 "${ISO[@]}" --save_plot --save_images
run steps_s1s2_cascade python test/eval_sample_steps.py --unet_ckpt "$S2M" --config "$C12" --npz_dir "$N1" --pad_size 1024 "${CAS[@]}" --save_plot --save_images

echo "===== [$(date '+%F %T')] ALL DONE — logs in $LOGDIR ====="
