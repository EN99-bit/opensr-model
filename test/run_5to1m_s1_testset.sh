#!/usr/bin/env bash
# Testset3600 eval for the 5→1m s1 stage-2 checkpoint (80/10/10 run), ISOLATED:
# stage-2 conditioned on the GT 5m aerial (--cond_5m_dir), no stage-1 in the loop.
#
# Mirrors the three scripts that populated the 5m_unet testset3600 folder, so the
# output folder gets the same layout (metrics.csv + cfg/cfgpp plots + images,
# metrics_steps.csv + steps_plots, metrics_oracle.csv + oracle_plots):
#   eval_cfg.py          -> CFG/CFG++ guidance sweep
#   eval_sample_steps.py -> DDIM step-count sweep (fixed gs=1)
#   eval_unet.py         -> x0-oracle diagnostic (single-step recon vs noise level)
#
# Run from the repo root with your python env active, e.g. in screen:
#   screen -S 5to1m   # then:
#   bash test/run_5to1m_s1_testset.sh 2>&1 | tee test/results-testset3600/_logs/5to1m_s1.log
#
# Each run logs to test/results-testset3600/_logs/<name>.log and FAILURES DO NOT
# STOP the rest. Runs are sequential on purpose — each one fans out across all
# visible GPUs, so running them concurrently would just contend.
set -u
cd "$(dirname "$0")/.."                       # repo root

LOGDIR=test/results-testset3600/_logs
mkdir -p "$LOGDIR"

# ── checkpoint / config / data ──────────────────────────────────────────────
CKPT=checkpoints/5to1m-s1-80-10-10/unet/unet5to1-s1-epoch=0221-val_loss=0.176917.ckpt
C1=opensr_model/configs/config_1m.yaml        # s1 variant (in_channels=10)
N1="$HOME/npz/apr2025/5to1m-test-stage2-only" # 1m test tiles (this model's seed-42 split)
N5="$HOME/npz/apr2025/5m-npz"                 # GT 5m aerial, looked up by tile-stem
OUT=test/results-testset3600
BS=1                                          # tiles/batch at pad 1024; drop to 2 if OOM

ISO=(--cond_5m_dir "$N5")

run () {  # run <logname> <command...>
  local name="$1"; shift
  echo "===== [$(date '+%F %T')] START $name ====="
  if "$@" > "$LOGDIR/$name.log" 2>&1; then
    echo "===== [$(date '+%F %T')] DONE  $name ====="
  else
    echo "===== [$(date '+%F %T')] FAIL  $name  (rc=$?, see $LOGDIR/$name.log) ====="
  fi
}

# ── evals ───────────────────────────────────────────────────────────────────
run cfg_5to1m_s1    python test/eval_cfg.py          --unet_ckpt "$CKPT" --config "$C1" --npz_dir "$N1" --pad_size 1024 "${ISO[@]}" --sampling_steps 100 --batch_size "$BS" --out_dir "$OUT" --save_plot --cfgpp --save_images
run steps_5to1m_s1  python test/eval_sample_steps.py --unet_ckpt "$CKPT" --config "$C1" --npz_dir "$N1" --pad_size 1024 "${ISO[@]}" --gs 1.0 --batch_size "$BS" --out_dir "$OUT" --save_plot --save_images
run oracle_5to1m_s1 python test/eval_unet.py         --unet_ckpt "$CKPT" --config "$C1" --npz_dir "$N1" --pad_size 1024 "${ISO[@]}" --out_dir "$OUT" --save_plot --save_images

echo "All done -> $OUT/5to1m-s1-80-10-10_unet5to1-s1-epoch=0221-val_loss=0.176917__cond5m/"
