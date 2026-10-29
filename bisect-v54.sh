#!/usr/bin/env bash
# bisect-v54.sh <label>:<recipe> [<label>:<recipe> ...]
# Each experiment swaps recipe.yaml, boots the TP=2 cluster with ./run.sh under a deadline, captures
# the container's exit reason (docker inspect, before anything removes it) and a 2 s host-memory
# timeline, then tears both boxes down. Always ends by restoring the v5.3 lane and probing a real
# completion — unless an experiment ends healthy AND KEEP=1 is set, in which case that lane is kept.
# Detached by the caller (setsid nohup ... &); every attempt logs to bisect-<ts>.log.
set -uo pipefail
cd "$(dirname "$0")"
TS=$(date +%Y%m%d-%H%M%S)
LOG=$PWD/bisect-$TS.log
exec >>"$LOG" 2>&1
RECIPE_BAK=recipe.yaml.bak-v5.3-pre-v54-port-20261029-081840
RUN_BAK=run.sh.bak-v5.3-lane
NAME=qwen38-flash-next-cluster
W=claidler@192.168.86.214
DEADLINE=${DEADLINE:-900}
KEEP=${KEEP:-0}

say(){ echo "$(date -u '+%H:%M:%SZ') $*"; }
probe(){   # a real completion, not just /health
  local nonce="zzyzx-$RANDOM"
  curl -sf --max-time 240 http://127.0.0.1:8000/v1/chat/completions \
    -H 'content-type: application/json' \
    -d "{\"model\":\"qwen-3.8-next\",\"messages\":[{\"role\":\"user\",\"content\":\"Repeat exactly this token and nothing else: $nonce\"}],\"max_tokens\":512,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    | grep -q "$nonce"
}

capture(){  # capture <label> — everything about how it died, BEFORE stop.sh removes the container
  local label=$1
  say "── $label: container state ──"
  docker inspect "$NAME" --format 'head   status={{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} err={{.State.Error}} started={{.State.StartedAt}} finished={{.State.FinishedAt}}' 2>&1
  ssh_w "docker inspect '$NAME' --format 'worker status={{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} err={{.State.Error}} started={{.State.StartedAt}} finished={{.State.FinishedAt}}'" 2>&1
  say "── $label: head log tail ──"
  docker logs --tail 40 "$NAME" 2>&1 | tr '\r' '\n' | grep -v '^Compiling CuTeDSL kernels: *[0-9]*%' | tail -30 | cut -c1-400
  say "── $label: head last warmup progress ──"
  docker logs --tail 400 "$NAME" 2>&1 | tr '\r' '\n' | grep -o 'Compiling CuTeDSL kernels: *[0-9]*%|[^,]*' | tail -1
  say "── $label: worker log tail ──"
  ssh_w "docker logs --tail 40 '$NAME'" 2>&1 | tr '\r' '\n' | grep -v '^Compiling CuTeDSL kernels: *[0-9]*%' | tail -20 | cut -c1-400
  say "── $label: head memory ──"
  free -m
  for f in /sys/fs/cgroup/system.slice/docker-*.scope/memory.stat; do
    [ -f "$f" ] && { echo "cgroup $(dirname "$f" | xargs basename):"; grep -E '^(anon|file|swapcached)' "$f"; }
  done 2>/dev/null
  say "── $label: head kernel ──"
  sudo -n dmesg -T 2>/dev/null | grep -iE 'oom|killed process|nvrm|xid|segfault|traps:' | tail -15
  say "── $label: worker kernel ──"
  ssh_w "sudo -n dmesg -T 2>/dev/null | grep -iE 'oom|killed process|nvrm|xid|segfault|traps:' | tail -15" 2>&1
  say "── $label: head journal (docker) ──"
  journalctl -u docker --since "-20 min" --no-pager 2>/dev/null | grep -iE 'oom|kill|error|died|exit' | tail -15
}

run_exp(){  # run_exp <label> <recipe> → 0 if run.sh reports healthy
  local label=$1 recipe=$2
  say "=== experiment $label ← $recipe"
  cp -f "$recipe" recipe.yaml
  local memf=$PWD/bisect-mem-$label-$TS.log
  ( while :; do
      printf '%s ' "$(date -u +%H:%M:%S)"
      free -m | awk '/^Mem:/{printf "used=%s free=%s avail=%s ",$3,$4,$7} /^Swap:/{printf "swapused=%s ",$3}'
      printf 'cgroup=%s ' "$(cat /sys/fs/cgroup/system.slice/docker-*.scope/memory.current 2>/dev/null | head -1)"
      echo
      sleep 2
    done ) >"$memf" 2>&1 &
  local smp=$!
  local t0; t0=$(date +%s)
  timeout "$DEADLINE" ./run.sh
  local rc=$? dt=$(( $(date +%s) - t0 ))
  kill "$smp" 2>/dev/null; wait "$smp" 2>/dev/null
  say "$label: run.sh rc=$rc after ${dt}s · memory timeline $(basename "$memf")"
  say "$label: peak memory"
  awk '{for(i=1;i<=NF;i++){if($i~/^used=/){gsub(/used=/,"",$i); if($i+0>m)m=$i+0}}} END{print "  peak used "m" MB"}' "$memf" 2>/dev/null
  capture "$label"
  ./stop.sh || true
  wait_mem 100 180 || true
  return $rc
}

revert(){
  say "REVERT to the v5.3 lane"
  ./stop.sh || true
  cp -f "$RECIPE_BAK" recipe.yaml
  cp -f "$RUN_BAK" run.sh
  if timeout 1200 ./run.sh && probe; then say "v5.3 lane is back and serving"; else say "v5.3 lane ALSO unhealthy — needs a human"; fi
}

say "bisect start · deadline ${DEADLINE}s/experiment · keep=$KEEP"
for spec in "$@"; do
  label=${spec%%:*}; recipe=${spec#*:}
  if run_exp "$label" "$recipe"; then
    if probe; then
      if [ "$KEEP" = 1 ]; then
        say "=== $label is healthy and serving — KEEPING it (no revert)"
        exit 0
      fi
      say "=== $label is healthy and serving — will revert to v5.3 as instructed"
    else
      say "=== $label: run.sh healthy but the completion probe failed"
    fi
    break
  fi
done
revert
say "bisect finished"
