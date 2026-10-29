#!/usr/bin/env python3
"""30-minute soak for the qwen-3.8-next lane.

Every cycle: one single-stream thinking-on request (default 512 tokens), then a
health probe, then sleep so the cycle is ~--interval seconds. Records per-cycle
decode tok/s, acceptance from /metrics deltas, and health. Writes JSON lines to
--out and a final summary. Designed to run detached (setsid nohup).
"""
import argparse, json, statistics, time, urllib.request, urllib.error

BASE = "http://127.0.0.1:8000"
PROMPT = ("Explain, carefully and in detail, how a modern distributed inference "
          "server splits a mixture-of-experts model across two GPUs, including "
          "expert parallelism, all-to-all communication, and load balancing.")


def metric(name):
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=5) as r:
            for line in r:
                s = line.decode("utf-8", "replace")
                if s.startswith("vllm:" + name + "{"):
                    return float(s.rsplit(" ", 1)[1])
    except Exception:
        pass
    return None


def health():
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
            return r.status
    except Exception:
        return 0


def request(max_tokens, thinking=True, timeout=600):
    body = {
        "model": "qwen-3.8-next",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.time(); ttft = None; last = None; n = 0; usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                    now = time.time()
                    if ttft is None:
                        ttft = now - t0
                    n += 1; last = now
    total = time.time() - t0
    ct = (usage or {}).get("completion_tokens") or n
    dec = (ct - 1) / (last - (t0 + ttft)) if ttft and last and last > t0 + ttft else 0.0
    return {"ttft_ms": round((ttft or 0) * 1000), "completion_tokens": ct,
            "wall_s": round(total, 2), "decode_tok_s": round(dec, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--interval", type=float, default=60)
    ap.add_argument("--out", default="soak.jsonl")
    a = ap.parse_args()
    end = time.time() + a.minutes * 60
    rows = []
    with open(a.out, "w") as f:
        while time.time() < end:
            t = time.time()
            d0 = metric("spec_decode_num_draft_tokens_total")
            a0 = metric("spec_decode_num_accepted_tokens_total")
            g0 = metric("generation_tokens_total")
            row = {"t": time.strftime("%H:%M:%S"), "health": health()}
            try:
                row.update(request(a.max_tokens))
            except Exception as e:
                row["error"] = str(e)[:200]
            d1 = metric("spec_decode_num_draft_tokens_total")
            a1 = metric("spec_decode_num_accepted_tokens_total")
            g1 = metric("generation_tokens_total")
            if None not in (d0, a0, d1, a1) and d1 > d0:
                steps = (d1 - d0) / 7.0
                row["accepted_per_step"] = round((a1 - a0) / steps, 2)
                row["tok_per_step"] = round((g1 - g0) / steps, 2)
            rows.append(row)
            f.write(json.dumps(row) + "\n"); f.flush()
            time.sleep(max(0, a.interval - (time.time() - t)))
    ok = [r["decode_tok_s"] for r in rows if "decode_tok_s" in r and r.get("health") == 200]
    bad = [r for r in rows if "error" in r or r.get("health") != 200]
    summary = {"cycles": len(rows), "healthy_cycles": len(ok), "failed_cycles": len(bad),
               "median_decode_tok_s": round(statistics.median(ok), 1) if ok else 0,
               "min_decode_tok_s": min(ok) if ok else 0,
               "max_decode_tok_s": max(ok) if ok else 0,
               "accepted_per_step_median": (statistics.median(
                   [r["accepted_per_step"] for r in rows if "accepted_per_step" in r])
                   if any("accepted_per_step" in r for r in rows) else None)}
    print(json.dumps({"summary": summary}), flush=True)
    with open(a.out + ".summary", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
