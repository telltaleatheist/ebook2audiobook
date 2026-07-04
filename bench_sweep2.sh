#!/bin/zsh
# Round 2: continuous vs chunk at realistic length, width-128 knee probe,
# cache-limit A/B. Logs to bench_results2.log
PY="/Users/telltale/Library/Application Support/BookForge/runtime/e2a-env/bin/python"
cd /Users/telltale/Projects/ebook2audiobook-latest
LOG=bench_results2.log
: > $LOG
run() {
  echo "=== $* ===" | tee -a $LOG
  "$PY" bench_orpheus_mlx.py "$@" 2>&1 | grep --line-buffered -v "Fetching\|deprecated" | tee -a $LOG
}
# long-run chunk vs continuous at width 96 (refill effect needs length)
run --mode chunk --width 96 --n 288
run --mode continuous --width 96 --n 288 --decode-batch
# knee probe at 128
run --mode chunk --width 128 --n 288
run --mode continuous --width 128 --n 288 --decode-batch
# cache-limit A/B: does bounding the cache to 8 GB cost throughput?
run --mode continuous --width 96 --n 96 --decode-batch --cache-limit-gb 8
run --mode chunk --width 96 --n 96 --no-clear-cache --cache-limit-gb 8
echo "SWEEP2 COMPLETE" | tee -a $LOG
