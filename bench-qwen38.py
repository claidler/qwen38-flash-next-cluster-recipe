#!/usr/bin/env python3
"""Streaming benchmark for the qwen-3.8-next lane. Stdlib only.

Usage:
  bench.py --reps 3 --max-tokens 1024 --thinking on
  bench.py --reps 1 --max-tokens 512 --thinking on --concurrency 32

Reports per rep: ttft_ms, completion_tokens, decode_tok_s (inter-chunk),
wall_tok_s (completion_tokens / total wall). Prints a JSON line per rep and a
summary line with median decode_tok_s.
"""
import argparse, json, statistics, sys, threading, time, urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
PROMPT = ("Write a detailed, careful explanation of how a modern GPU executes a "
          "matrix multiplication, covering tiling, shared memory, tensor cores, "
          "and memory bandwidth. Be thorough and precise.")


def one(rep, max_tokens, thinking, prompt, timeout=600):
    body = {
        "model": "qwen-3.8-next",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    ttft = None
    last = None
    chunks = 0
    usage = None
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
                txt = d.get("content") or d.get("reasoning") or d.get("reasoning_content") or ""
                if txt:
                    now = time.time()
                    if ttft is None:
                        ttft = now - t0
                    chunks += 1
                    last = now
    end = time.time()
    total = end - t0
    ct = (usage or {}).get("completion_tokens") or chunks
    dec = (ct - 1) / (last - (t0 + ttft)) if ttft and last and last > t0 + ttft else 0.0
    return {
        "rep": rep, "thinking": thinking, "ttft_ms": round((ttft or 0) * 1000),
        "completion_tokens": ct, "wall_s": round(total, 2),
        "decode_tok_s": round(dec, 1), "wall_tok_s": round(ct / total, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--thinking", choices=["on", "off"], default="on")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--prompt", default=PROMPT)
    a = ap.parse_args()
    think = a.thinking == "on"
    rows = []
    if a.concurrency == 1:
        for i in range(a.reps):
            try:
                row = one(i + 1, a.max_tokens, think, a.prompt)
            except Exception as e:
                print(json.dumps({"rep": i + 1, "error": str(e)}), flush=True)
                continue
            rows.append(row)
            print(json.dumps(row), flush=True)
    else:
        lock = threading.Lock()
        threads = []
        for i in range(a.concurrency):
            def run(idx=i + 1):
                try:
                    row = one(idx, a.max_tokens, think, a.prompt)
                except Exception as e:
                    row = {"rep": idx, "error": str(e)}
                with lock:
                    rows.append(row)
                    print(json.dumps(row), flush=True)
            t = threading.Thread(target=run)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    ok = [r["decode_tok_s"] for r in rows if "decode_tok_s" in r]
    if ok:
        print(json.dumps({"summary": True, "thinking": a.thinking,
                          "concurrency": a.concurrency, "n": len(ok),
                          "median_decode_tok_s": round(statistics.median(ok), 1),
                          "min": min(ok), "max": max(ok)}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
