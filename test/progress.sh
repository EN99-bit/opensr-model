#!/usr/bin/env bash
# At-a-glance status for run_testset3600.sh.
#   bash test/progress.sh        # one-shot snapshot
#   watch -n10 bash test/progress.sh   # live, refreshing every 10s
#
# Shows: how many of the jobs are done (with their wall times), and for any
# job still running, the latest tqdm line (%/tiles/ETA) from its log.
cd "$(dirname "$0")/.."
OUT=test/results-testset3600
LOGDIR="$OUT/_logs"
TIMINGS="$LOGDIR/timings.csv"
ALL=(oracle sample_steps cfg op_nohm op_hm no_s1_op ldsr2 ldsr2_hm bicubic bicubic_hm)

done_jobs=$(tail -n +2 "$TIMINGS" 2>/dev/null | cut -d, -f1)
ndone=$(printf '%s\n' "$done_jobs" | grep -c .)

echo "── completed ($ndone / ${#ALL[@]}) ──"
[ -f "$TIMINGS" ] && column -t -s, "$TIMINGS" || echo "  (no timings yet — not started?)"

echo
echo "── running ──"
any=0
for j in "${ALL[@]}"; do
  printf '%s\n' "$done_jobs" | grep -qx "$j" && continue   # already finished
  [ -f "$LOGDIR/$j.log" ] || continue                       # not started yet
  any=1
  # tqdm writes \r-separated updates; grab the newest progress line
  last=$(tail -c 4000 "$LOGDIR/$j.log" | tr '\r' '\n' | grep -E '%\|| it/s|tiles' | tail -n1)
  printf "  %-13s %s\n" "$j" "${last:-<loading model…>}"
done
[ "$any" -eq 0 ] && echo "  (none active)"

echo
echo "── pending ──"
pend=""
for j in "${ALL[@]}"; do
  printf '%s\n' "$done_jobs" | grep -qx "$j" && continue
  [ -f "$LOGDIR/$j.log" ] && continue
  pend="$pend $j"
done
echo "  ${pend:-(none)}"
