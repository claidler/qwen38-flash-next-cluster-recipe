# Qwen3.8-Flash-Next two-box lane — v5.4 port result

**Date:** 2026-10-29
**Lane:** `v5.4-lane` @ `efc3133` (3 commits ahead of `origin/v5.4-lane`, **not pushed**)
**Serving now:** `http://192.168.86.179:8000/v1`, model id `qwen-3.8-next`, healthy on both boxes.

---

## 1. Bottom line

Upstream recipe v5.4 (`5a59faa`) is landed as **k4a = full v5.4 with `num_speculative_tokens: 7`**.

| measurement (c=1, thinking on) | v5.3 | full v5.4 (K=12) | **k4a (K=7)** |
|---|---|---|---|
| generic prompt, 1024 tok, n=3 | **70.2** | 68.4 | **71.4** |
| code prompt, 512 tok, n=2 | 81.9 | 78.3 | — |
| 30-cycle soak, 512 tok, median | not run | 65.3 | **64.0** |
| c=32, 256 tok, per stream | 19.1 | 20.9 | **21.6** |
| c=32 aggregate | ~611 | ~669 | **~691** |
| 8k-token agent turn TTFT | — | — | **cold 3.55 s / warm 0.16 s** |
| accepted per draft | 0.205 | 0.106 | **0.216** |

**Net:** single-stream c=1 is +1.7% over v5.3 (within rep noise), batched c=32 is **+13%**, and
draft economy is back to v5.3 levels instead of half. Stability is proven: 30/30 soak cycles
healthy on both boxes, 288 s boot, CuTeDSL 52/52, warm-up OAK 5.0 s.

**The 142.9 tok/s DONE-MEANS figure is not reproducible and is treated as a bad number** — see §5.
Against the *measured* v5.3 baseline (70.2), the lane meets its target. Against 142.9 it does not,
and nothing on either recipe has ever produced 142.9 on this checkpoint.

---

## 2. What actually blocked v5.4 (not the image)

v5.4 appeared to die at CuTeDSL unit 39/52. It never was an image problem.

**Root cause: both GB10s were latched in the low-power state.**

| | SM clock | power | 45 s bf16 burn |
|---|---|---|---|
| head, latched | 669 MHz | 15.6 W | 30.7 TFLOPS |
| worker, latched | 656 MHz | 14.1 W | **14.9 TFLOPS** |
| head, healthy | 2255 MHz | 87.5 W | **92.8 TFLOPS** |
| worker, healthy | 2262 MHz | 88.5 W | **91.6 TFLOPS** |

Symptoms: ~650–700 MHz, ~15 W, `clocks_throttle_reasons.active = 0x0` (no reason given),
`nvidia-smi -lgc` accepted but silently ignored, survives reboot. Only a **full power drain**
(wall + brick + unit unplugged ≥ 5 min, reconnect wall → brick → unit) cleared it.

Once the clocks were healthy, **unpatched full v5.4 booted 52/52 in 316 s** with every required
line. The `patches/mbx-warmup-trace.patch` from the earlier attempt is therefore **not needed**
and was not applied. The 39/52 death was the warmup taking ~3× longer than the boot watchdog
allowed while the GPU ran at 1/3 clock.

**Prevention:** `/tmp/gb10-check.py` (45 s burn) reports HEALTHY / LATCHED per box; run it after
any unexplained slowdown or reboot before blaming the recipe.

---

## 3. Per-knob A/B results

All at c=1, thinking on, 1024 tokens, 3–4 reps, GB10s verified at full clock (2405–2522 MHz, no
throttle reason). Each variant differs from k4a in exactly one knob.

| knob | variant | tok/s | vs k4a | verdict |
|---|---|---|---|---|
| skinny GEMM plans | `MBX_SKINNY_GEMM_SM12X: "0"` | 48.5 | **−32%** | **+47% with it** — biggest lever |
| RoCE all-reduce | `VLLM_ENABLE_ROCE_ALLREDUCE: "0"` | 65.6 | −8.1% | **+8.8% with it** |
| draft head | `MBX_DRAFT_HEAD: "0"` | 66.8 | −6.4% | **+6.9% with it** |
| draft depth | K=12 instead of 7 | 68.4 | −4.2% | **K=7 wins** (see below) |
| MXFP8 b12x | kernel + config removed | 70.9 | −0.7% | **neutral**, kept |
| PLE prefill ckpt | `MBX_PLE_PREFILL_CKPT: "0"` | 68.4 | −4.2% | **neutral** on TTFT: 3.69/0.15 vs 3.55/0.16 — kept |

Notes:
- The k8 (no-skinny) run spread 41.8–57.1 tok/s. Clocks were re-checked immediately after:
  2405 MHz, no throttle, so that spread is JIT/content variance, not a clock dip.
- PLE prefill checkpoints: the warm 0.15–0.16 s hit comes from **prefix caching**, not from the
  checkpoints — turning them off changed nothing. Kept because it is upstream default and costs
  nothing measurable.
- MXFP8 b12x: 70.9 vs 71.4 is inside the ±5–10% rep noise. Kept rather than disabled.
- `VLLM_ENABLE_ROCE_ALLREDUCE` is logged by vLLM as an unknown env var, but the image's own
  `b12x_roce_all_reduce.py` honors it: the `RoCEnante all-reduce is live` line appears with it and
  vanishes without it.

### K=12 → K=7

| | K=12 | K=7 |
|---|---|---|
| bench c=1, 1024 tok (n=3) | 68.4 | **71.4** (+4.4%) |
| soak median, 512 tok (n=30) | 65.3 | 64.0 (−2.0%) |
| soak mean | 63.9 | 63.0 |
| soak p10 / p90 | 61.3 / 69.7 | 60.6 / 67.9 |
| accepted per draft | 0.106 | **0.216** |
| accepted per step | 0.81 | **1.34** |
| c=32 per stream | 20.9 | **21.6** |

Bench and soak disagree by ~6 points, in opposite directions, with overlapping distributions —
i.e. **c=1 throughput is a wash**. K=7 is chosen because it buys the same wall clock for **half the
draft compute**, and it is better at c=32. K=12 has no measured upside on this checkpoint.

---

## 4. Final config (what changed vs v5.3)

- image: `cjjlaidler/qwen38-flash-next-vllm:v5.3` → `myllmbox/qwen38-flash-next-cluster-vllm:v5.4`
  (digest `sha256:46cd09cd…`, the two-box image)
- `MBX_SKINNY_GEMM_SM12X: "1"`, `VLLM_ENABLE_ROCE_ALLREDUCE: "1"`, `B12X_ROCE_CACHE_DIR`
- `MBX_DRAFT_HEAD: "1"`, two-stage draft basis auto-detected by `run.sh` from the image
- `MBX_DEPTH_MODE: dynamic`, `min 3`, ceiling `[(2,9),(4,8),(6,7),(12,6),(14,5)]`, sync-free on
- `speculative-config` K **7** (upstream ships 12)
- bf16 KV, `prefix-match-unit: 64`, PLE prefill checkpoints, MXFP8 b12x linear backend
- `run.sh`: v5.4 port (draft-basis autodetect, no compressed-tensors unfuse, no depth-json write)
- kept untouched: checkpoint `claidler/…-LMHead-NVFP4-GDNMSE`, `served-model-name: qwen-3.8-next`,
  `host: 0.0.0.0`, `moe-backend: humming`, 400k YaRN overrides, warm-up + OAK self-check

Verified on the live lane:

```
RoCEnante all-reduce is live: first routed all-reduce is 163840 bytes (bfloat16); NCCL remains the fallback above 2MB.
mbx depth: dynamic · min 3 · max 7 (K 7; drafted-depth ceiling [(2, 9), (4, 8), (6, 7), (12, 6), (14, 5)]) · window 48 · ...
cutedsl_warmup.py:107] Warming up CuTeDSL compile_units=52
✓ warm-up: 5.0 s · The word is **OAK**, and its middle letter is **A**.
```

---

## 5. The 142.9 tok/s criterion

I could not reproduce 142.9 on either recipe, at any prompt, at any concurrency, on healthy clocks:

| prompt / mode | tok/s |
|---|---|
| generic thinking-on, c=1, 1024 tok (v5.3) | 70.2 |
| generic thinking-on, c=1, 1024 tok (k4a) | 71.4 |
| code prompt, c=1, 512 tok (v5.3) | 81.9 |
| "repeat ab 256×", c=1, 512 tok | 80.1 |
| "list 1..300", c=1, 512 tok | **119.6** |
| c=2 aggregate | ~102 |
| c=32 aggregate | ~691 |

Upstream's own published numbers for this release are 53.3 → 83.8 single-user thinking-on, and the
earlier A/B report on this box measured 71.9. 142.9 is ≈2× every single-stream number observed;
it looks like an aggregate or a different metric mislabelled as single-stream. It is recorded here
as **not a valid target** rather than quietly dropped.

Consequence: the "30-min soak ≥ 142.9" gate cannot be met by construction. The gate used instead:
30/30 healthy cycles + no throughput regression vs the measured v5.3 baseline. Both hold.

**One measurement still open:** no v5.3 soak was run at the 512-token/60 s soak protocol, so the
soak numbers (65.3 / 64.0) have no v5.3 counterpart. The v5.3 comparison is bench-to-bench
(70.2 vs 71.4) plus c=32 (611 vs 691). Closing that would cost one v5.3 boot + 30 min + one boot
back; not done because it takes the agent's own serve down twice for a number that cannot change
the decision.

---

## 6. Commits

```
efc3133 run.sh: stop writing the dead cache/mbx-depth.json (no reader in the v5.4 image)
4f5226e recipe: record the measured A/B verdict on every v5.4 knob (comments only)
5748372 v5.4 lane: land k4a (upstream v5.4 + K=7) — the measured winner
```

`5748372` holds the **exact bytes that were booted and soaked** (recipe + run.sh). `efc3133`
removes a dead config writer that wrote `mode: "off"` to `cache/mbx-depth.json` on every boot
(nothing in the image reads that file — grep over `/opt` and `/usr/local/lib` finds no reference;
depth is env-driven). It is a one-commit revert if the next boot disagrees.

**Not pushed, not merged to main** — as required, pending a two-box boot + 30 min healthy. That
condition is now met, so pushing is the user's call.

---

## 7. Rollback

```bash
cd ~/qwen38-flash-next-cluster-recipe
git revert --no-edit efc3133     # only if the run.sh cleanup misbehaves
# full v5.3:
cp recipe.yaml.bak-v5.3-pre-v54-port-20261029-081840 recipe.yaml
cp run.sh.bak-v5.3-lane run.sh
./run.sh
```

Preserved untouched: `recipe.yaml.bak-v5.3-pre-v54-port-20261029-081840`, `run.sh.bak-v5.3-lane`,
`recipe.yaml.v54-minimal`, `recipe.yaml.bisect-v54-full`, `recipe.yaml.ab-k*`, all bisect/soak logs.

---

## 8. Operational notes

- **Boot is 5–6 min** at full clock (288 s for the final boot); CuTeDSL warmup 17 s.
- The pi's `qwen-kit-watchdog.timer` is **running again** (was stopped during the work). It needs 2
  consecutive health failures, a 30-min gap, and skips while `run.sh` is already running. Its dry
  run just now cleared the fail counter without kicking. It accepts `qwen-3.8-next` as a served id,
  so the rename pin is safe.
- If throughput halves with no throttle reason: **check the clocks first** (`/tmp/gb10-check.py`),
  then power-drain. Do not touch the recipe.
- Soak: `python3 soak-qwen38.py --minutes 30 --max-tokens 512 --interval 60 --out soak-<ts>.jsonl`
- Bench: `python3 bench-qwen38.py --reps 3 --max-tokens 1024 --thinking on`
- TTFT: `python3 ttft-qwen38.py` (cold ~3.55 s, warm ~0.16 s on ~8k tokens)
