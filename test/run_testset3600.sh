#!/usr/bin/env bash
# Full re-evaluation of the 5m model on the CLEAN 3600-tile test set.
# Replaces the old 21-tile `5m-untouched` runs. Everything (images + numbers)
# is persisted under test/results-testset3600/ so this never has to be rerun.
#
# Scope: full sweeps (cfg, sample-steps, oracle) for the S1+S2 model + bicubic &
#        LDSR-S2 baselines (with/without histogram matching) + operating-point
#        (gs=1, 100 steps) runs via eval.py for S1+S2 (HM + no-HM) and the no-S1
#        model (for the fusion ablation tab:fusion). All with opensr-test metrics,
#        --save_plot/--save_images ON. (VAE ceiling deferred — see memory
#        thesis-3600-testset-eval.)
#
# Run in screen/tmux from repo root:
#   screen -S testset3600
#   bash test/run_testset3600.sh 2>&1 | tee test/results-testset3600/_logs/run.log
#
# Failures DO NOT stop the batch; each job logs to _logs/<name>.log.
set -u
cd "$(dirname "$0")/.."                      # repo root

OUT=test/results-testset3600
LOGDIR="$OUT/_logs"
mkdir -p "$LOGDIR"

# ── checkpoint / config / data ──────────────────────────────────────────────
M5=checkpoints/5m/unet-no-latents/unet-epoch=0098-val_loss=0.102384.ckpt   # 5m e98 (S1+S2)
C10=opensr_model/configs/config_10m.yaml
M_NOS1="checkpoints/5m/unet-no-s1-matched/unet-no-s1-epoch=0079-val_loss=0.111461.ckpt"  # 5m no-S1 (S2 only) e79
C_NOS1=opensr_model/configs/config_10m_no_s1.yaml
NPZ="$HOME/npz/apr2025/5m-test-split"                                       # 3600 clean test tiles

TIMINGS="$LOGDIR/timings.csv"
[ -f "$TIMINGS" ] || echo "job,status,seconds,hms,started,finished" > "$TIMINGS"
BATCH_START=$(date +%s)

hms () { printf '%02d:%02d:%02d' $(( $1/3600 )) $(( ($1%3600)/60 )) $(( $1%60 )); }

run () {  # run <logname> <command...>
  local name="$1"; shift
  local t0 t1 dt started status
  t0=$(date +%s); started=$(date '+%F %T')
  echo "===== [$started] START $name ====="
  if "$@" > "$LOGDIR/$name.log" 2>&1; then status=DONE; else status="FAIL(rc=$?)"; fi
  t1=$(date +%s); dt=$(( t1 - t0 ))
  echo "===== [$(date '+%F %T')] $status $name  (elapsed $(hms $dt), see $LOGDIR/$name.log) ====="
  echo "$name,$status,$dt,$(hms $dt),$started,$(date '+%F %T')" >> "$TIMINGS"
}

# Operating point for all non-sweep model runs: cfg OFF (guidance 1.0), 100 steps.
GS_OP=1.0
STEPS_OP=100

# ══════════════════════════════════════════════════════════════════════════════
# Phase A — 4-GPU sharded sweeps (run sequentially; each uses all 4 GPUs)
# ══════════════════════════════════════════════════════════════════════════════
run oracle        python test/eval_unet.py --unet_ckpt "$M5" --config "$C10" \
                    --npz_dir "$NPZ" --pad_size 256 --out_dir "$OUT" --save_plot --save_images

run sample_steps  python test/eval_sample_steps.py --unet_ckpt "$M5" --config "$C10" \
                    --npz_dir "$NPZ" --pad_size 256 --gs "$GS_OP" --batch_size 8 \
                    --out_dir "$OUT" --save_plot --save_images

# heaviest (~7.6 h at batch_size 8): full guidance + cfg++ sweep at 100 steps
run cfg           python test/eval_cfg.py --unet_ckpt "$M5" --config "$C10" \
                    --npz_dir "$NPZ" --pad_size 256 --sampling_steps "$STEPS_OP" --batch_size 8 \
                    --out_dir "$OUT" --save_plot --cfgpp --save_images

# ══════════════════════════════════════════════════════════════════════════════
# Phase B — single-GPU jobs (eval.py & baselines don't shard) run CONCURRENTLY,
#   one pinned per GPU. 7 jobs over 4 GPUs → two waves (~7 h total).
#   eval.py computes opensr-test + PSNR/SSIM/LPIPS automatically; do NOT pass
#   --opensr (that swaps in the official HuggingFace model and ignores --ckpt).
#   op_nohm (S1+S2) and no_s1_op (S2-only) are same-script at gs=1 → directly
#   comparable for the fusion ablation table (tab:fusion).
# ══════════════════════════════════════════════════════════════════════════════
# wave 1 (GPUs 0-3): the long single-GPU jobs
run op_nohm   env CUDA_VISIBLE_DEVICES=0 python test/eval.py --ckpt "$M5" --config "$C10" --npz_dir "$NPZ" \
                --steps "$STEPS_OP" --guidance "$GS_OP" --out_dir "$OUT" &
run op_hm     env CUDA_VISIBLE_DEVICES=1 python test/eval.py --ckpt "$M5" --config "$C10" --npz_dir "$NPZ" \
                --steps "$STEPS_OP" --guidance "$GS_OP" --histogram_match --out_dir "$OUT" &
run no_s1_op  env CUDA_VISIBLE_DEVICES=2 python test/eval.py --ckpt "$M_NOS1" --config "$C_NOS1" --npz_dir "$NPZ" \
                --steps "$STEPS_OP" --guidance "$GS_OP" --no_s1 --out_dir "$OUT" &
run ldsr2     env CUDA_VISIBLE_DEVICES=3 python test/eval_ldsr2.py --npz_dir "$NPZ" \
                --out_csv "$OUT/ldsr2_baseline/metrics.csv" --save_images &
wait

# wave 2 (GPUs 0-2): ldsr2 histmatch + cheap bicubic baselines
run ldsr2_hm   env CUDA_VISIBLE_DEVICES=0 python test/eval_ldsr2.py --npz_dir "$NPZ" --histogram_match \
                --out_csv "$OUT/ldsr2_baseline_histmatch/metrics.csv" --save_images &
run bicubic    env CUDA_VISIBLE_DEVICES=1 python test/eval_bicubic.py --npz_dir "$NPZ" \
                --out_csv "$OUT/bicubic_baseline/metrics.csv" --save_images &
run bicubic_hm env CUDA_VISIBLE_DEVICES=2 python test/eval_bicubic.py --npz_dir "$NPZ" --histogram_match \
                --out_csv "$OUT/bicubic_baseline_histmatch/metrics.csv" --save_images &
wait

echo "===== [$(date '+%F %T')] ALL DONE — outputs in $OUT, logs in $LOGDIR ====="
echo "===== TOTAL wall: $(hms $(( $(date +%s) - BATCH_START ))) — per-job breakdown in $TIMINGS ====="
column -t -s, "$TIMINGS"
