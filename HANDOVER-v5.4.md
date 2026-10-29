# Task: land the good parts of upstream v5.4 on my Qwen3.8-Flash-Next cluster recipe

## Where things live

- Two DGX Sparks (GB10, aarch64): head `spark-9f70` (192.168.86.179), worker `spark-91de`
  (192.168.86.214). SSH as `claidler`, key auth, both reachable from this machine.
- Recipe dir on the head: `~/qwen38-flash-next-cluster-recipe` — my fork of
  `github.com/claidler/qwen38-flash-next-cluster-recipe`, with the original repo added as the
  `upstream` remote (`github.com/myllmbox/qwen38-flash-next-cluster-recipe`, at tag/commit `v5.4`,
  `5a59faa`).
- `./run.sh` composes and launches the container on both boxes, waits for `/health`, runs a
  warm-up, and prints the OpenAI endpoint. `./stop.sh` stops both. Boot takes ~10 minutes.
- Live model served at `http://192.168.86.179:8000/v1` as `qwen-3.8-next`.
  **My own agent runs on that serve — if it goes down, I go down with it.**
- A/B harness and prior results: `~/qwen38-moe-ab-report.md` (that is how `moe-backend: humming`
  was chosen: −9.4% TTFT, +0.07 accept length vs marlin).

## What I am trying to do

Upstream shipped v5.4 of this recipe with large perf gains (their published single-user
thinking-on throughput 53.3 → 83.8 tok/s, mixed c=32 49.0 → 65.5, 128k prefill 3,049 tok/s,
cached-128k TTFT 0.35 s). I want those gains **without** wrecking the lane I serve on. I am not
after an exact re-run of upstream's config: I keep my own checkpoint
(`claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4-GDNMSE`), so their numbers are a
target, not a promise, and each knob gets judged here.

## What is already done — do not redo it

1. **Diffed** my recipe against `upstream/main`; classified the v5.4 changes into
   port-now / needs-image-rebase / leave-alone.
2. **Found that I was never on the two-box image.** My `image:` pointed at
   `cjjlaidler/qwen38-flash-next-vllm:v5.3`, whose image ID is identical to the *solo*
   (one-box) `myllmbox/qwen38-flash-next-vllm:v5.3`. Grepped both images: `VLLM_ENABLE_ROCE_ALLREDUCE`
   and `MBX_SKINNY_GEMM_SM12X` are **absent** there, so my RoCE all-reduce and TP=2 skinny-GEMM
   env lines have been inert. They exist in `myllmbox/qwen38-flash-next-cluster-vllm:v5.3` and `:v5.4`.
3. **Pulled `myllmbox/qwen38-flash-next-cluster-vllm:v5.4` on both boxes** (digest
   `sha256:46cd09cdcd96ef69887df91d6e546cb43a02991ae2642a358930aec52418bfc8`) and verified in the
   image that `MBX_DEPTH_CEIL` (in `mbx_depth.cpython-312-aarch64.so`), `MBX_DEPTH_AGREE_INIT`,
   `MBX_PLE_PREFILL_CKPT` (`/opt/mbx/patches/58-ple-prefill-checkpoints.py`), `MBX_DRAFT_HEAD`,
   the b12x RoCE all-reduce and the TP=2 skinny plans are all present.
4. **Branch `v5.4-lane`** (commit `e2b15d8`, pushed to my origin) holds the full v5.4 port:
   cluster image; depth policy moved from the `mtp_depth:` YAML block to `env: MBX_DEPTH_*`
   (`AGREE_INIT=1`, `MODE=dynamic`, `MIN=3`, `WINDOW=48`, `PROMOTE=60,45`, `DEMOTE=25,15`,
   `CEIL="[2 -> 9, 4 -> 8, 6 -> 7, 12 -> 6, 14 -> 5]"`); `num_speculative_tokens` 7 → 12;
   bf16 KV + `mamba-ssm-cache-dtype: bfloat16`; `kv-cache-memory` 36G → 34G; `max-num-seqs` 64 → 32;
   `block-size` 3312 → 3456; `prefix-match-unit: 64`; `MBX_PLE_PREFILL_CKPT=1`; `MBX_DRAFT_HEAD=1`;
   `PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True`; MXFP8 onto b12x
   (`kernel-config: '{"linear_backend_per_quant":{"mxfp8":"b12x"}}'`, `MBX_MXFP8_A16=0`,
   `B12xMxfp8LinearKernel` dropped from `VLLM_DISABLED_KERNELS`).
   `run.sh` there is upstream v5.4's **plus my warm-up block re-applied** (the only diff vs
   upstream is that one hunk) — v5.4 deleted the `cache/mbx-depth.json` writer and the warm-up.
5. **Two cutover attempts, both self-reverting, both failed.** Scripts
   `cutover-v5.4.sh` and `cutover-v5.4b.sh` (in the recipe dir) stop, boot, probe with a real
   completion, and on failure or a >20 min boot copy `recipe.yaml.bak-v5.3-pre-v54-port-20261029-081840`
   and `run.sh.bak-v5.3-lane` back and relaunch. Logs: `cutover-v5.4-20261029-083342.log`,
   `cutover-v5.4b-20261029-090647.log`, `worker-v5.4b-20261029-090647.log` (the round-2 script
   streams the worker's `docker logs` to that file, which is how the failure was captured).
   - Attempt 1 (full port) reached `mbx depth: dynamic · min 3 · max 12 (K 12; ceiling [(2,9),(4,8),(6,7),(12,6),(14,5)])`
     and `mbx draft head: the head is already quantized (torch.int32) — the drafter shares it`,
     then the head container exited; the worker was already gone.
   - Attempt 2 (`recipe.yaml.v54-minimal`: cluster image with **everything else at my proven v5.3
     values** — fp8 KV, 36G, 64 seats, K=7, block-size 3312) failed the same way at ~8.5 min.
6. **The current lead, and where I want you to start.** In both failures the last thing either box
   logged was:
   `cutedsl_warmup.py:107] Warming up CuTeDSL compile_units=52 names=('Qwen4Exp HC down+SiLU GEMM', 'shape-dynamic skinny GEMM')`
   — on the head *and* the worker, ~30 s before the containers died. My working v5.3 lane logs the
   same warmup with `compile_units=34` and survives. No Python traceback, no host OOM in `dmesg`,
   no NVRM/Xid lines: a hard kill. Also confirmed working when it got that far:
   `RoCEnante all-reduce is live: first routed all-reduce is 327680 bytes (bfloat16); NCCL remains
   the fallback above 2MB` — the two-box layer we could never light up before.

## What I want from you

Bisect the CuTeDSL-warmup death of the v5.4 cluster image and get it serving, one variable at a
time, cheapest suspicion first:

1. Reproduce the boot deliberately and capture the **exit reason**: `docker inspect` the dead
   container for `ExitCode`/`OOMKilled`, `journalctl -u docker -n 200`, and
   `nvidia-smi -q -d PAGE_MIGRATION,ECC` / `dmesg` on *both* boxes around the death.
2. First cut, one variable: `MBX_SKINNY_GEMM_SM12X: "0"` (the new 52-unit plan table is the prime
   suspect). If that boots, re-enable it and try narrowing instead (`MBX_SKINNY_GEMM_LIMIT=128,256`).
   Then, one at a time, the other cluster-image-only knobs: `VLLM_ENABLE_ROCE_ALLREDUCE=0`
   (isolates the b12x RoCE path), `MBX_DRAFT_HEAD`, `MBX_PLE_PREFILL_CKPT`, `block-size 3456` +
   `prefix-match-unit 64`, bf16 KV + K=12, and the b12x MXFP8 trio.
3. Keep the loop fast: prefer booting the head-only container (`--tensor-parallel-size 1` on one
   box) or a short `--max-model-len` to reach the warmup faster, then do the full two-box boot once
   you believe it fits.

## Rules

- **Never leave the boxes in a state where nothing serves.** Every attempt goes through a
  cutover script that reverts to the v5.3 backups on failure or a >20 min boot. Run attempts
  detached (`setsid nohup … &`) so an dropped SSH does not kill them, and log to a timestamped
  file. The v5.3 lane is live now — `cjjlaidler/qwen38-flash-next-vllm:v5.3`, `/health` 200.
- Do not delete or overwrite the backups: `recipe.yaml.bak-v5.3-pre-v54-port-20261029-081840`,
  `run.sh.bak-v5.3-lane` (and keep `recipe.yaml.v54-minimal` as the bisection base).
- Keep these mine, whatever the config does: `model:` my Ablit/GDNMSE checkpoint,
  `served-model-name: qwen-3.8-next` (pinned — changing it kick-loops my watchdog),
  `host: 0.0.0.0` (LAN access), `moe-backend: humming` (measured win, upstream still ships marlin),
  the 400k YaRN `hf-overrides`, and the warm-up + OAK self-check block in `run.sh`.
- Work on branch `v5.4-lane`; commit each step with the reason in the message. Do not merge to
  `main` or push until a two-box boot serves for 30 minutes with acceptance and RoCE lines healthy.
- Judge, do not assume: use the A/B harness in `~/qwen38-moe-ab-report.md` (single-user thinking-on
  and code prompts at c=1 and c=32, accept length from `/metrics` or `./view.sh`) before keeping any
  knob. A knob that does not measurably help stays commented out with a note saying so.

## Done means

`./run.sh` on the v5.4 cluster image comes up healthy on both boxes with the warm-up passing,
`RoCEnante all-reduce is live` and the `mbx depth: dynamic … ceiling [...]` lines in the log, and
a 30-minute soak shows my thinking-on c=1 throughput at or above the v5.3 baseline
(142.9 tok/s single-stream on the current lane). Report the deltas per knob — upstream's numbers
are not mine until measured here.
