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

So the 10-28 "humming is the right kernel for GB10" intuition is real but narrow: **humming wins prefill/TTFT (+15% prefill tok/s, −0.4 s cold 8k TTFT), marlin wins decode at c=1 and decisively at c=4–8**.

## Realistic-turn test (turn-c4.py): c=4, each turn a unique cold ~13k prompt, thinking-on, 1024 out

| backend | median wall/turn | median TTFT | median decode/stream | aggregate decode |
|---|---|---|---|---|
| marlin | **37.4 s** | 10.0 s | **38.4** | **~156 tok/s** |
| humming | 37.6 s | **9.1 s** | 36.5 | ~146 tok/s |

**Overall verdict: marlin.** Turn wall-time is a wash at c=4 and marlin wins aggregate (+7%) and c=1/c=8 decode; humming's only edge is ~1 s of TTFT per cold turn, which matters only for prefill-dominated turns (very long prompt, short output) — the long-context profile, out of scope here. Lane left on marlin. For a TTFT-first profile, humming is the rollback:

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

## Flare ideas (sxuff/qwen38-flash-next-flare, X @filicroval 2116295184421888436) — A/B'd 2026-10-31

Engine swap rejected (single-box Rust runtime, stock checkpoint, our 2-box numbers already 2–4× it; full assessment: `skills/dgx-spark-llm-inference/references/sxuff-flare-recipe-2026-10-31.md`). The two transferable ideas were A/B'd from the pristine base (marlin/K=7/8192):

| variant | c=4 turn wall | c=4 TTFT | c=4 agg decode | c=8 /stream | soak | verdict |
|---|---|---|---|---|---|---|
| base (8192, K=7) | 37.4 s | 10.0 s | ~156 | 37.0–38.5 | 18/18 | reference |
| max-num-batched-tokens 4096 ("prefill priority" analogue) | 38.7 s (+3.5%) | 12.3 s (worse) | ~158 | 37.5 | 18/18 med 66.3 | **not promoted** — smaller chunks hurt TTFT, Flare's priority-16 has no vLLM equivalent |
| K=3 (their fixed-3-drafts) | boot FAILED (head died at startup, TCPStore broken pipe on rank1; no OOM, clocks healthy 2145 MHz) | — | — | — | — | **not promoted**; K=7 stands |

Spec drafting at c=8 confirmed active on our lane (accepted/draft ≈ 0.2; capture sizes to 192 rows, depth ceiling trims instead of disabling) — the Flare "8-row graph limit" does not apply here. Greedy and mixed/chunked prefill already on. Lane restored to marlin/K=7/8192, health 200.

## Sweep 2 (2026-10-31, second round of knobs)

| variant | result | verdict |
|---|---|---|
| `moe-backend: flashinfer_cutedsl` | boot FAILS (WorkerProc init error) at healthy clocks — same class as b12x: MoE backend incompatible with this model on TP=2 | **ruled out** |
| fp8kv / depthmin2 / ncclstripe / prefix128 | not run — both GB10s latched at 208 MHz mid-sweep; run.sh preflight (added this session) aborted every boot with the LATCHED message instead of phantom-crashing | **blocked on hardware**, rerun after power drain |

Next-round candidates found in the image's env surface: `VLLM_ROCE_ALLREDUCE_MAX_SIZE` (default 2MB — raise to route bigger c=8 collectives over RoCE instead of NCCL), `MBX_PLE_ASYNC*` / `MBX_FUSED_DRAFT` (untested), plus `flashinfer_cutedsl_batched`/`flashinfer_trtllm` MoE backends (likely same failure class as cutedsl).

## Sweep 2 results (2026-10-31, on f1 QSFP ports after the power event; base re-measured same session — all variants ≈ base, NONE promoted)

| variant | c=1 tok/s | c=8 /stream | c=4 wall / TTFT | cold 8k TTFT | soak | verdict |
|---|---|---|---|---|---|---|
| base2 (marlin/K7/8192, f1) | 69.7 | 33.1 | 46.0 s / 14.8 s | 5.79 s | 18/18 med 66.2 | reference (current env) |
| fp8kv | 70.6 | 32.8 | 45.8 / 14.7 | 5.70 | 18/18 med 65.3 | not promoted |
| depthmin2 | 69.6 | 32.8 | 47.1 / 14.8 | 5.77 | 18/18 med 64.4 | not promoted |
| ncclstripe (MERGE_NICS=0, QPS=4, SPLIT=1) | 70.3 | 32.9 | 47.2 / 14.8 | 5.78 | 18/18 med 65.8 | not promoted |
| prefix128 | 70.1 | 32.8 | 47.1 / 14.8 | 5.78 | 18/18 med 66.9 | not promoted |
| cutedsl (earlier, healthy clocks) | boot FAILS | — | — | — | — | ruled out |
| K=3 (earlier) | boot FAILED | — | — | — | — | ruled out |

**Environmental regression since the 2026-10-31 morning power events:** same base config measured 2026-10-30/31 morning (f0 ports): c=8 37–38.5, c=4 wall 37.4 s / TTFT 10.0 s, cold TTFT 3.96 s, 52k prefill ~3110 tok/s. After the power events (QSFP reseated into **f1** ports; netplan+cluster.env switched f0→f1, backups `cluster.env.bak-f0-20261031`): c=8 33.1, c=4 wall 46 s / TTFT 14.8 s, cold TTFT 5.79 s, 52k prefill ~2190 tok/s. c=1 unchanged (~70). dmesg logs `insufficient power on the PCIe slot (27W)` ×4/box (power brick on an extender cable); PCIe link itself is full 32GT/s x4; GPU burn HEALTHY (91 TFLOPS) on both. **Fix: brick direct into the wall (no extender), full drain both boxes, then re-measure base.**

**run.sh fixes landed:** preflight burn file pushed to worker as /tmp/gb10chk.py (repo not mirrored there — empty burn caused false LATCHED abort); SIB_PROBE empty result no longer kills the boot under `set -e` (CX7 port without IPv4 now warns and stays on one rail).

**Fabric root cause found & fixed (2026-10-31 ~12:00):** after the power events, the CX7 logged `insufficient power on the PCIe slot (27W)` and `ib_write_bw` measured only **~1.5 GB/s** (healthy: 22–24 GB/s per the playbook) — that degraded all c≥4/prefill numbers (sweep2 variants were all measured in this state; their relative verdicts vs base2 still hold). Moving the QSFP cables back to the **f0** ports + a plain reboot cleared the dmesg errors; ib_write_bw recovered to **13.3 GB/s** (4MB, 4 QP) and base3 re-measured at morning levels (c=1 72.6, c=8 37.8, c=4 wall 37.5/35.4 s, TTFT 10.1/10.0, cold 3.93 s, 52k prefill 3.2k tok/s). Power/extender unchanged. **Do not move the QSFP cables to the f1 sockets or power-cycle without a full drain — f0 + clean boot is the known-good state.** netplan/cluster.env back on f0 (`cluster.env.bak-f0-20261031` kept).

## Sweep 4 (2026-10-31, HEALTHY fabric after f0+reboot fix; base3 = marlin/K7/8192: c=1 72.6, c=8 37.8, c=4 wall 37.5/35.4 s TTFT 10.1/10.0, cold TTFT 3.93, 52k prefill 3.2k, soak 18/18)

| variant | c=1 | c=8 | c=4 wall | cold TTFT | soak | verdict |
|---|---|---|---|---|---|---|
| base3 | 72.6 | 37.8 | 36.5 s | 3.93 s | med 66-67 | reference |
| `VLLM_ROCE_ALLREDUCE_MAX_SIZE: 8MB` (roce8m; log confirms "fallback above 8MB") | 71.0 | 38.0 | 36.5 s | 3.92 s | 18/18 med 67.5 | not promoted (c≤8 collectives already ≤2MB) |
| ncclstripe (MERGE_NICS=0, QPS=4, SPLIT=1) | 74.9 | 38.2 | 36.6 s | 3.88 s | — | not promoted (within noise; keep MERGE_NICS=1) |
| depthmin2 (MBX_DEPTH_MIN 3→2) | 70.8 | 37.0 | 36.9 s | 3.93 s | 18/18 med 65.2 | not promoted (−2.5% c=1) |

## Sweep 5/6 (2026-10-31, GitHub-sweep queue; healthy fabric; base3: c=1 72.6, c=8 37.8, c=4 wall 36.5 s/TTFT 10.0, cold TTFT 3.93)

RoCE gate first: `ib_write_bw` both f0 HCAs = **13.3 GB/s each**, dmesg `insufficient power` count = **0** — healthy, benchmarks valid.

| variant | result | verdict |
|---|---|---|
| eugr env block (NCCL_MAX_CTAS=8, BUFFSIZE=1MB, AOT, V2 runner, SSM DS, MXFP8_LM_HEAD) | boot FAILS — `CUDA error: misaligned address` | not portable from eugr's b12x image to ours |
| eugr transport only (NCCL_MAX_CTAS=8, NCCL_BUFFSIZE=1MB) | boot FAILS — CUDA error | ruled out |
| `VLLM_USE_V2_MODEL_RUNNER=1` alone | c=1 72.9, c=8 37.3, c=4 36.5 s | neutral — not promoted |
| `VLLM_USE_AOT_COMPILE=1` alone | c=1 72.6, c=8 37.2, c=4 38.0 s | neutral — not promoted |
| `VLLM_SSM_CONV_STATE_LAYOUT=DS` alone | c=1 70.4 (−3%), c=8 37.8 | not promoted |
| `--api-server-count 2` | boot FAILS — `cannot be used with --headless` (worker) | ruled out for TP=2 |
| DFlash2 drafter (method dflash, incoai 0.6B, K=8) | boot FAILS — `NotImplementedError: Qwen4Exp speculative decoding supports only native MTP and linear n-gram` | ruled out on this model/image |
| salient-data `skinny-gemm-rows` patch (QWEN38_SKINNY_FILL) via `server.patches` | patch APPLIES cleanly but head dies after model load (patch written for a different vLLM base) | ruled out on v5.4 |

Audit notes: RecoverSSM (#58863) and NVFP4 draft lm_head are already live in v5.4 (boot logs show both). `VLLM_INDEXER_FP8` absent from image. GDN-MSE is already our checkpoint (`...-GDNMSE`). Remaining real upside per the sweep: checkpoint swaps (azampatti FP4-Local-AZ, hybrid48-uncensored — both need download + quality gauntlet) and ops hygiene only.

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
