#!/bin/zsh
# Orpheus MLX batch-width sweep — logs to bench_results.log
PY="/Users/telltale/Library/Application Support/BookForge/runtime/e2a-env/bin/python"
cd /Users/telltale/Projects/ebook2audiobook-latest
LOG=bench_results.log
: > $LOG
run() {
  echo "=== $* ===" | tee -a $LOG
  "$PY" bench_orpheus_mlx.py "$@" 2>&1 | grep -v "Fetching\|deprecated" | tee -a $LOG
}
# chunk mode (current production behavior) across widths
for w in 16 32 48 64 96; do
  run --mode chunk --width $w --n 96
done
# cache-clear cadence A/B at width 48
run --mode chunk --width 48 --n 96 --no-clear-cache
# continuous batching (hypothesized fix)
run --mode continuous --width 48 --n 96 --decode-batch
run --mode continuous --width 96 --n 96 --decode-batch
# per-sentence reference
run --mode single --n 8
echo "SWEEP COMPLETE" | tee -a $LOG
