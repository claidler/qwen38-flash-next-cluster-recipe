#!/usr/bin/env bash
# Round 2: boot the v5.4 CLUSTER image with everything else left at the proven v5.3 values,
# so a failure points at the image and not at our tuning. Self-reverting; runs detached, because
# this serve is what runs the agent that wrote it.
set -uo pipefail
cd "$(dirname "$0")"
TS=$(date +%Y%m%d-%H%M%S)
LOG=$PWD/cutover-v5.4b-$TS.log
WLOG=$PWD/worker-v5.4b-$TS.log
exec >>"$LOG" 2>&1
RECIPE_BAK=recipe.yaml.bak-v5.3-pre-v54-port-20261029-081840
RUN_BAK=run.sh.bak-v5.3-lane
V54_RECIPE=recipe.yaml.v54-minimal
NAME=qwen38-flash-next-cluster
W=claidler@192.168.86.214
BOOT_DEADLINE=1200

say(){ echo "$(date -u '+%H:%M:%SZ') $*"; }
probe(){
  local nonce="zzyzx-$RANDOM"
  curl -sf --max-time 240 http://127.0.0.1:8000/v1/chat/completions \
    -H 'content-type: application/json' \
    -d "{\"model\":\"qwen-3.8-next\",\"messages\":[{\"role\":\"user\",\"content\":\"Repeat exactly this token and nothing else: $nonce\"}],\"max_tokens\":512,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    | grep -q "$nonce"
}
worker_dump(){   # why did the worker die — the container may already be gone, so also the kernel
  say "worker: last lines of $WLOG and dmesg"
  [ -s "$WLOG" ] && tail -60 "$WLOG"
  ssh -o ConnectTimeout=8 $W "dmesg -T 2>/dev/null | grep -iE 'oom|killed process|nvrm|xid' | tail -12" 2>&1 | tail -12
}
revert(){
  say "REVERT to the v5.3 lane"
  ./stop.sh || true
  cp -f "$RECIPE_BAK" recipe.yaml
  cp -f "$RUN_BAK" run.sh
  if timeout "$BOOT_DEADLINE" ./run.sh && probe; then say "v5.3 lane is back and serving"; else say "v5.3 lane ALSO unhealthy — needs a human"; fi
}

say "round 2 start: v5.4 cluster image, v5.3 tuning otherwise"
cp -f "$RECIPE_BAK" recipe.yaml.bak-pre-v54b-$TS       # what was live before this attempt
cp -f "$V54_RECIPE" recipe.yaml
git show e2b15d8:run.sh > run.sh.new && chmod +x run.sh.new && mv -f run.sh.new run.sh   # v5.4 run.sh + our warm-up
: > "$WLOG"
( while :; do ssh -o ConnectTimeout=8 $W "docker logs --tail 400 $NAME 2>&1" >> "$WLOG" 2>/dev/null; sleep 15; done ) &
STREAM=$!
trap 'kill "$STREAM" 2>/dev/null || true' EXIT

./stop.sh || true
if timeout "$BOOT_DEADLINE" ./run.sh; then
  say "run.sh says healthy; probing a real completion"
  for i in 1 2 3; do
    if probe; then kill "$STREAM" 2>/dev/null; say "CUTOVER OK — v5.4 cluster image live: $(grep -m1 'image:' recipe.yaml | tr -s ' ')"; exit 0; fi
    say "probe $i failed; retrying in 20s"; sleep 20
  done
  say "healthy but no usable completion"
else
  say "boot failed or exceeded ${BOOT_DEADLINE}s"
fi
worker_dump
revert
kill "$STREAM" 2>/dev/null || true
say "round 2 finished"
