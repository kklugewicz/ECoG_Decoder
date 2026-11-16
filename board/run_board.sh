#!/bin/bash
# Full board study, in the order the paper ran it:
#   accuracy -> sweep -> soak -> paced (sleep-wait) -> paced (spin-wait)
#
# Run ON the board from the repository root, after copying board/inputs/
# (made by board/dump_inputs.py on the host):
#     PY=~/et/bin/python bash board/run_board.sh
# PY is the Python that has the ExecuTorch runtime (default: python3).
#
# A watchdog kills the benchmark if the SoC passes 95 C, and each stage waits
# for the SoC to cool to 50 C so stages start from comparable temperatures.
# For the paper: performance governor, heatsink + fan, desktop session idle,
# suspend masked (sudo systemctl mask sleep.target suspend.target).
cd "$(dirname "$0")/.."
PY=${PY:-python3}
R=board/results
mkdir -p $R
S=$R/status.txt
temp() { echo $(( $(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0) / 1000 )); }
echo "start $(date) governor $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u) temp $(temp)C" > $S

( while true; do
    t=$(temp)
    if [ "$t" -ge 95 ]; then
      echo "WATCHDOG: ${t}C at $(date), killing benchmark" >> $S
      pkill -f board_bench.py
    fi
    sleep 2
  done ) &
WD=$!

cool() {
  for i in $(seq 1 60); do
    [ "$(temp)" -le 50 ] && break
    sleep 5
  done
  echo "  stage start temp $(temp)C" >> $S
}

stage() {   # stage <log name> <board_bench args...>
  local log=$1; shift
  cool
  $PY -u board/board_bench.py "$@" --out $R > $R/$log.log 2>&1
  echo "$* exit $? $(date)" >> $S
  sync
}

stage accuracy   accuracy
stage sweep      sweep
stage soak       soak --soak-s 600
stage paced_sleep paced
stage paced_spin paced --spin
kill $WD
echo "done $(date)" >> $S
