# Qwen3.8-Flash-Next on Two DGX Sparks

### Two-node DGX Spark cluster recipe for Qwen3.8-Flash-Next NVFP4-QAD (`cjjlaidler/qwen38-flash-next-cluster-vllm:v1`)

Serves [`claidler/Qwen3.8-Flash-Next-NVFP4-MXFP8-QAD-LMHead-NVFP4-GDNMSE`](https://huggingface.co/claidler/Qwen3.8-Flash-Next-NVFP4-MXFP8-QAD-LMHead-NVFP4-GDNMSE) (the non-abliterated QAD body with NVFP4 PLE embedding table and repacked NVFP4 output head + the retrained GDN-MSE MTP drafter — normal refusal behaviour, +8.6% decode). For the abliterated body (no refusals, no guardrails — research / red-teaming / private use behind your own moderation), swap `model:` in `recipe.yaml` to [`claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4-GDNMSE`](https://huggingface.co/claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4-GDNMSE).

Two rules for model swaps:
- **Only ever change `model:`** — leave `served-model-name:` (`qwen-3.8-flash-next`) alone so client configs survive the swap. Changing the served name per-checkpoint is how cutovers 404.
- **Both bodies use plain expert scales and load on `:v1` today** — the non-ablit model is built from the plain-scale `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` checkpoint. (The earlier `...-MXFP8-CSF-QAD-LMHead-NVFP4` twin was accidentally built from the CSF sibling base, which vLLM 0.30 cannot load; it is superseded.)

Two DGX Sparks (GB10, 119 GB unified memory each), TP=2 over ConnectX RoCE v2 RDMA link. High-throughput speculative decoding (dynamic-depth MTP: drafts 3-5 tokens per step from per-batch acceptance, block verification).

## Quick Start

```bash
git clone https://github.com/claidler/qwen38-flash-next-cluster-recipe.git
cd qwen38-flash-next-cluster-recipe
./run.sh
```

- **First run:** Automatically sets up the cluster (`./setup.sh`), discovers the ConnectX interconnect, probes RDMA GID tables, downloads model weights to `./models`, syncs to worker rank 1, and serves on port `8000`.
- `./stop.sh` stops both nodes cleanly.
- `./view.sh` shows real-time cluster stats and verifies RDMA packet counters.

### Test Inference

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-3.8-flash-next",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

---

## Included Runtime Patches

This repository includes custom runtime patches mounted directly into the vLLM container (`cjjlaidler/qwen38-flash-next-cluster-vllm:v1`):

1. **`ngram_embedding.py`**
   - Implements NVFP4 quantized PLE (predictive lookup / n-gram embedding) table handling.
   - Handles 128 packed NVFP4 PLE embedding shards (`shard_<k>.weight`, `shard_<k>.weight_scale`, `weight_scale_2`).
   - Automatically supports row-sharded ETP=2 GPU distribution across ranks.

2. **`flashinfer_mxfp8.py`**
   - Patches `FlashInferCutlassMxfp8LinearKernel` to pad weights with $N < 128$ (such as $N=48$ linear attention gates `in_proj_ba`) up to hardware 128-row tile alignment during load.
   - Transparently slices outputs back to original dimensions ($N=48$) during forward GEMM operations.

---

## Measured Performance (2× DGX Spark, RoCE v2 RDMA, K=5 MTP)

| Concurrent Requests | Peak tok/s | Avg tok/s | Per-stream tok/s | Acceptance Rate (K=5) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | **121** | 107 | 107 | ~87% (5.33 / 6) |
| 2 | **185** | 170 | 85 | ~85% |
| 4 | **275** | 260 | 65 | ~86% |
| 8 | **395** | 380 | 47 | ~85% |
| 16 | **520** | 495 | 31 | ~84% |
| 32 | **644** | 610 | 19 | ~85% |
| 64 | **808** | 760 | 12 | ~83% |

- Speculative drafting acceptance rate averages 85-87% on code and complex instructions.
- 41 GB pinned KV cache per node provides over 2.5 million pooled KV tokens in BF16 (split across the 2 nodes).

---

## Memory Management & Stability

Unified memory on the DGX Spark GB10 requires careful page handling to avoid driver stalls:

1. **Memory Availability Gate:** `run.sh` waits until both nodes report $\ge 100\text{ GB}$ free before launching.
2. **Page Cache Eviction:** Evicts checkpoint files from the page cache prior to container initialization using non-privileged POSIX cache drops.
3. **Kernel Memory Compaction:** Run `./tune-host.sh` with sudo privileges on both nodes to set `vm.compaction_proactiveness=0`, preventing background compaction stalls during active decoding.

---

## Architecture & Configuration

All model serving configuration is managed in [`recipe.yaml`](recipe.yaml):
- **Image:** `cjjlaidler/qwen38-flash-next-cluster-vllm:v1` (manifest digest `sha256:697d4364853312f3b50354020eb99cb12585228c9d16d1288261edb9631fa93b`, vLLM 0.30.0 + the b12x fabric stack)
- **Model:** `claidler/Qwen3.8-Flash-Next-NVFP4-MXFP8-QAD-LMHead-NVFP4-GDNMSE` (default, non-abliterated, plain scales, loads on `:v1`) — or the abliterated variant `claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4-GDNMSE`
- **Served model name:** `qwen-3.8-flash-next` — generic and identical across model swaps; do not change it per-checkpoint
- **CPUSet:** Cores `5-9,15-19` (GB10 Cortex-X925 3.9 GHz performance cores)
- **KV Cache Memory:** `41000000000` (41 GB per box pinned)
- **MoE Backend:** Marlin NVFP4 W4A16
- **Speculative Config:** MTP K=5 probabilistic draft with block verification

Node-specific network interfaces, RDMA HCAs, and interconnect IPs are stored in `cluster.env` (generated by `setup.sh` and not committed).

---

## Image history & staged A/B arms

`v1` is the current image — published under the `cjjlaidler` namespace, retagged from the original
v5.2 build (manifest digest `sha256:697d4364853312f3b50354020eb99cb12585228c9d16d1288261edb9631fa93b`,
image ID `4210dc4befb5`). The tag is immutable by convention: any change to the image ships a new
tag + digest. It is vLLM 0.30.0 for GB10 (SM121a) with the b12x
fabric/spec stack; the NVFP4 load path (n-gram table + 4-bit output head) is baked in. A/B evidence 2026-10-10:
quality trio 3/3, needle 3/3 at 120k, decode neutral vs fixed K=5 — kept for low-acceptance
requests that stop wasting draft compute.

**Dynamic draft depth is ACTIVE** via `cache/mbx-depth.json` (gitignored, hot-reloaded by mtime —
edit it, no restart):

```json
{"mode":"dynamic","min":3,"max":7,"window":48,"promote":[60,45],"demote":[25,15]}
```

- **`mode` must be the literal string `dynamic`** — anything else (including `"on"`) disables the
  feature and drafts silently fall back to fixed K. The boot log line `mbx depth: dynamic ...`
  (grep `docker logs` for `mbx depth`) is the only proof it is live.
- The effective ceiling is the compiled `num_speculative_tokens` (5 here): the config's `max: 7`
  clamps to 5, and the log line prints the clamped value. Raise `max` above 5 only together with
  a K bump in `recipe.yaml` (needs a restart + new CUDA graphs).
- `"log":true` adds per-decision lines (depth transitions + acceptance %) — use for one
  measurement pass, then drop it.
- `"mode":"off"` = fixed-K behaviour, hot-swappable for an A/B.

Remaining staged arms (one variable at a time, gate = quality trio 3/3 + needle 3/3 AND decode
>5% or c=16/64 aggregate >10%):

| Arm | Enable | What it does |
|:---|:---|:---|
| One-shot RoCE all-reduce | `env:` `VLLM_ENABLE_ROCE_ALLREDUCE: "1"` | b12x one-shot RDMA path (`b12x.comm.roce` — import-verified in v1) for cross-node TP collectives <= 2 MB instead of NCCL; fails its health check -> back to NCCL, never corrupts |
| Dual-rail NCCL | `cluster.env`: `HEAD_HCA`/`WORKER_HCA` = `rocep1s0f0,roceP2p1s0f0` | both ConnectX HCAs are up and ping-verified on this pair (0.1 ms RTT); NCCL stripes across both; prove bytes on the second rail via `port_xmit_data` counters |

The GB10 skinny-GEMM table (`MBX_SKINNY_GEMM_SM12X=1`) is **TP=1 only** — no effect here at TP=2.
SGLang was A/B'd against this recipe on 2026-08-30 and closed: 73.4 vs 84.9 tok/s aggregate at
c=3 — vLLM stays the serving engine.
