#!/usr/bin/env bash
# One-shot self-reverting v5.4 cutover for the Spark pair.
# Runs detached on purpose: this serve is what runs the agent that wrote it, so
# the agent is offline for the duration and cannot babysit the boot.
set -uo pipefail
cd "$(dirname "$0")"
LOG=$PWD/cutover-v5.4-$(date +%Y%m%d-%H%M%S).log
exec >>"$LOG" 2>&1
RECIPE_BAK=recipe.yaml.bak-v5.3-pre-v54-port-20261029-081840
RUN_BAK=run.sh.bak-v5.3-lane
BOOT_DEADLINE=1200                      # 20 min: past that, the boot is not coming back

say(){ echo "$(date -u '+%H:%M:%SZ') $*"; }

probe(){   # a real completion, not just /health returning 200
  local nonce="zzyzx-$RANDOM"
  curl -sf --max-time 240 http://127.0.0.1:8000/v1/chat/completions \
    -H 'content-type: application/json' \
    -d "{\"model\":\"qwen-3.8-next\",\"messages\":[{\"role\":\"user\",\"content\":\"Repeat exactly this token and nothing else: $nonce\"}],\"max_tokens\":512,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    | grep -q "$nonce"
}

revert(){
  say "REVERT to the v5.3 lane ($RECIPE_BAK, $RUN_BAK)"
  ./stop.sh || true
  cp -f "$RECIPE_BAK" recipe.yaml
  cp -f "$RUN_BAK" run.sh
  if timeout "$BOOT_DEADLINE" ./run.sh && probe; then say "v5.3 lane is back and serving"; else say "v5.3 lane ALSO unhealthy — needs a human"; fi
}

say "cutover start. live config was: $(grep -m1 'image:' recipe.yaml | tr -s ' ')"
./stop.sh || true
if timeout "$BOOT_DEADLINE" ./run.sh; then
  say "run.sh says healthy; probing a real completion"
  for i in 1 2 3; do
    if probe; then say "CUTOVER OK — v5.4 live: $(grep -m1 'image:' recipe.yaml | tr -s ' ')"; exit 0; fi
    say "probe $i failed; retrying in 20s"; sleep 20
  done
  say "healthy but no usable completion"
else
  say "boot failed, or exceeded ${BOOT_DEADLINE}s"
fi
revert
say "cutover finished"
