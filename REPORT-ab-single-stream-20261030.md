# Single-stream / c<=8 A/B session — 2026-10-30 (head spark-9f70, worker spark-91de)

**Target profile (Chris):** single-stream tok/s first, 4–8 concurrent streams second, long-context capacity not a goal.
**Baseline (k4a, K=7, humming, measured live this session):** c=1 thinking-on 1024 tok n=3 median **70.4** tok/s (68.5/70.6/70.4); c=8 512 tok median **35.0**/stream (~280 agg); 8k TTFT cold **3.55 s** / warm **0.16 s**.
Both GB10s verified healthy before every boot (idle SM 2405/2411 MHz, throttle 0x0, driver 580.173.02).

## Verdict table (one knob per boot, ~5.5 min boots, 18-cycle screening soak each)

| variant | c=1 median tok/s | vs baseline | c=8 /stream | vs baseline | 8k TTFT cold/warm | soak | accepted/draft | verdict |
|---|---|---|---|---|---|---|---|---|
| baseline k4a (K=7, humming, batched 8192) | 70.4 | — | 35.0 | — | 3.55 / 0.16 | — | 0.216 | reference |
| K=5 | 68.9 | −2.1% | 36.0 | +2.9% | 3.55 / 0.17 | 18/18, med 65.3 | 0.275 | not promoted (c=1 target) |
| K=6 | 67.4 | −4.3% | 33.9 | −3.1% | 3.56 / 0.15 | 18/18, med 65.1 | 0.231 | not promoted |
| K=8 | 68.9 | −2.1% | 34.6 | −1.1% | 3.58 / 0.15 | interrupted by next boot | — | not promoted |
| batched 16384 (K=7) | 69.7 | −1.0% | 35.1 | +0.3% | **3.77** / 0.15 | 18/18, med 64.0 | 0.194 | not promoted (cold TTFT +6%) |
| **marlin (K=7, batched 8192)** | **73.7** | **+4.7%** | **37.5** | **+7.1%** | 3.98 / 0.15 | 18/18, med 67.2 | 0.190 | **PROMOTED** |

K sweep summary (c=1): K=5 68.9 · K=6 67.4 · **K=7 70.4** · K=8 68.9 · K=12 68.4 (2026-10-29) — K=7 confirmed as the peak.

## Decision

**`moe-backend: marlin` is live** (commit `bc57fc2`, restored after the head-to-head). It wins the target metric at c=1 (+4.7%, consistent with the 2026-10-28 moe-ab: 72.5 vs 71.9) and clearly at c=8 (+7.1–10%, ~300–308 agg tok/s).

## Prefill head-to-head (added after the fact — `prefill-qwen38.py`, same day, same harness, healthy clocks)

| metric | humming | marlin | winner |
|---|---|---|---|
| prefill ~13k cold | 3624–3672 tok/s | ~3160 tok/s (rep1 1141 = first-cold JIT) | **humming +15%** |
| prefill ~52k cold | ~3540 tok/s | ~3110 tok/s | **humming +14%** |
| warm TTFT 13k / 52k | 0.17 / 0.24 s | 0.25 / 0.43 s | humming |
| cold 8k TTFT | 3.58 s | 3.98 s | humming |
| c=1 decode n=5 | 70.0 | **72.0** | **marlin +2.9%** |
| c=8 /stream | 34.0 | **38.5** | **marlin +13%** |
| soak median (512 tok) | 63.8 | 67.5 | marlin |
| accepted/draft | 0.192 | 0.198 | tie |

So the 10-28 "humming is the right kernel for GB10" intuition is real but narrow: **humming wins prefill/TTFT (+15% prefill tok/s, −0.4 s cold 8k TTFT), marlin wins decode at c=1 and decisively at c=4–8**. For an agent turn (~8k prompt + ≥1k output) the two roughly cancel at c=1 (18.2 s vs 18.2 s wall) and marlin pulls ahead with concurrency and longer outputs; humming only wins prefill-dominated turns (very long prompt, short output) — i.e. the long-context profile, which is out of scope. Hence marlin for the single-stream/c≤8 profile. For a TTFT-first profile, humming is the rollback:

```bash
cp recipe.yaml.ab-marlin-* recipe.yaml   # humming
./run.sh
```

## Ops changes landed (phase 0, commits `a24c578`, `c63de89`)

- `run.sh` preflight: per-box driver version + idle SM clock logged; <1400 MHz triggers the 45 s bf16 burn (`gb10-check.py`, now committed in the repo) and aborts with power-drain instructions if not HEALTHY — the GB10 low-power latch no longer masquerades as a boot crash.
- `run.sh` host fallback 0.0.0.0, tracking upstream `bfed7b8` (histories unrelated; applied as one-liner; recipe.yaml already pins `host: 0.0.0.0`).

## Final validation on the promoted config (marlin lane, live)

- needle-gate-fixed 120k x3: **3/3 PASS** (in ≈ 214.8–215.1k real prompt tokens, out 80–99 tok, no garbage)
- c=1 n=5, 1024 tok, thinking on: median **72.0** tok/s (71.2–74.2) vs humming baseline 70.4 → **+2.3%**
- c=8, 512 tok: median **38.5**/stream (~308 aggregate) vs 35.0 → **+10%**
- 30-min soak (512 tok, 60 s interval): **30/30 healthy**, median 67.5 tok/s, accepted/draft 0.198
- health 200, head SM 2405 MHz / 11 W idle after soak — no latch, no throttle

## Not done (deliberately)

- fp8 KV (value was 2x KV pool — long context is out of scope; bf16 KV kept for acceptance).
- NCCL striping vs MERGE_NICS (matters above the 2 MB RoCE all-reduce threshold — c<=8 stays under it).
- MBX_DEPTH_* micro-sweep, hybrid48-uncensored checkpoint swap (needs quality gauntlet; memory headroom not needed while long-ctx is out of scope).
- Watch-only: v5.5 image / vLLM #53896 / b12x MoE (needs the SM121 guard fix) / DFlash2 drafter.

## Probe suite going forward

`bench-qwen38.py` (c=1 + --concurrency), `ttft-qwen38.py` (8k cold/warm), **`prefill-qwen38.py` (NEW: cold prefill tok/s at ~13k/~52k + warm)**, `soak-qwen38.py`, `needle-gate-fixed.py`. Every A/B gate now: c=1 n≥3, c=8, 8k TTFT, 13k/52k prefill, 18-cycle soak; final promotion adds needle gate + 30-min soak.

## Commits

```
bc57fc2 ab: moe-backend marlin — PROMOTED (+4.7% c=1, +7.1% c=8; cold TTFT +0.43 s accepted)
d1a50c0 ab: max-num-batched-tokens 16384 — not promoted, reverted
3eb09cf ab: restore K=7 (sweep winner)
3447f4b ab: K=8 — not promoted
86e78be ab: K=6 — not promoted
1662401 ab: K=5 — not promoted
c63de89 run.sh: host fallback 0.0.0.0 (tracks upstream bfed7b8)
a24c578 run.sh: GB10 latch preflight + driver log; gb10-check.py committed
```

Not pushed — pushing is the user's call.
