#!/usr/bin/env bash
# Machine-wide memory guard for the scratch_0905 ladder runs.
#
#   * "actual occupation" = MemTotal - MemAvailable from /proc/meminfo
#     (whole machine, everybody's processes, page cache excluded)
#   * sampled every second; a 10-sample moving average above LIMIT_GB
#     kills ONE process -- the largest-RSS member of OUR runs (main.py rungs
#     / ramulator2 workers launched for scratch_0905) -- then keeps watching.
#     Nothing outside our runs is ever touched.
#   * also logs the aggregate CPU of our runs (cores) and warns above CORE_CAP.
#
#   usage: mem_guard.sh <logfile>      (runs until killed; one line per event)
LOG=${1:?logfile}
LIMIT_GB=${LIMIT_GB:-700}
CORE_CAP=${CORE_CAP:-64}
WINDOW=10
declare -a samples=()
tick=0

our_procs() {   # pid rss_kb pcpu args  -- only our ladder processes
    ps -eo pid,rss,pcpu,args --no-headers | awk '
        /python3 main.py/ && /scratch_0905/ {print; next}
        /^ *[0-9]+ +[0-9]+ +[0-9.]+ +[^ ]*ramulator2 / {print; next}'
}

echo "$(date +%F_%T) guard start limit=${LIMIT_GB}GB window=${WINDOW}s core_cap=${CORE_CAP}" >> "$LOG"
while true; do
    total_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
    avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    used_gb=$(( (total_kb - avail_kb) / 1048576 ))
    samples+=("$used_gb")
    if [ ${#samples[@]} -gt $WINDOW ]; then samples=("${samples[@]:1}"); fi
    sum=0; for s in "${samples[@]}"; do sum=$((sum + s)); done
    avg=$((sum / ${#samples[@]}))
    tick=$((tick + 1))
    if [ $((tick % 30)) -eq 0 ]; then
        cores=$(our_procs | awk '{c += $3} END {printf "%.1f", c / 100}')
        ours_gb=$(our_procs | awk '{r += $2} END {printf "%.1f", r / 1048576}')
        echo "$(date +%F_%T) used=${used_gb}GB avg10=${avg}GB ours_rss=${ours_gb}GB ours_cores=${cores}" >> "$LOG"
        awk -v c="$cores" -v cap="$CORE_CAP" 'BEGIN {if (c > cap) exit 0; exit 1}' && \
            echo "$(date +%F_%T) WARN our runs use ${cores} cores > ${CORE_CAP}" >> "$LOG"
    fi
    if [ ${#samples[@]} -eq $WINDOW ] && [ "$avg" -gt "$LIMIT_GB" ]; then
        victim=$(our_procs | sort -k2 -n -r | head -1)
        if [ -n "$victim" ]; then
            vpid=$(echo "$victim" | awk '{print $1}')
            vrss=$(echo "$victim" | awk '{printf "%.1f", $2 / 1048576}')
            vargs=$(echo "$victim" | cut -c1-160)
            kill "$vpid" 2>/dev/null
            echo "$(date +%F_%T) KILL avg10=${avg}GB > ${LIMIT_GB}GB -> pid ${vpid} rss=${vrss}GB :: ${vargs}" >> "$LOG"
            samples=()          # restart the window after acting
            sleep 3
        else
            echo "$(date +%F_%T) OVER avg10=${avg}GB but none of our processes is running" >> "$LOG"
            samples=()
        fi
    fi
    sleep 1
done
