#!/usr/bin/env python3
"""Realistic agent-turn probe: N concurrent turns, each with a UNIQUE ~8k-token prompt
(cold prefill, like parallel agents on different tasks), thinking-on, 1024 output tokens.
Reports per-turn wall_s / ttft_s / decode_tok_s + medians. Stdlib only.
Usage: turn-c4.py [--concurrency 4] [--max-tokens 1024] [--reps 1]
"""
import argparse, json, statistics, threading, time, urllib.request, uuid

URL = "http://127.0.0.1:8000/v1/chat/completions"
PARA = ("The quick brown fox jumps over the lazy dog while the distributed inference "
        "server routes experts across two GPUs and keeps the KV cache warm. ")


def one(idx, max_tokens, paras=500):
    prompt = f"task-{uuid.uuid4()}\n" + PARA * paras
    body = {"model": "qwen-3.8-next",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": True}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.time(); ttft = None; last = None; usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
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
                    last = now
    end = time.time()
    ct = (usage or {}).get("completion_tokens") or 0
    dec = (ct - 1) / (last - (t0 + ttft)) if ttft and last and last > t0 + ttft else 0.0
    pt = (usage or {}).get("prompt_tokens", 0)
    return {"turn": idx, "prompt_tokens": pt, "ttft_s": round(ttft or 0, 2),
            "wall_s": round(end - t0, 2), "decode_tok_s": round(dec, 1),
            "prefill_tok_s": round(pt / ttft) if ttft else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    for rep in range(a.reps):
        rows, lock, threads = [], threading.Lock(), []
        for i in range(a.concurrency):
            def run(idx=i + 1):
                try:
                    row = one(idx, a.max_tokens)
                except Exception as e:
                    row = {"turn": idx, "error": str(e)[:120]}
                with lock:
                    rows.append(row)
                    print(json.dumps(row), flush=True)
            t = threading.Thread(target=run)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        ok = [r for r in rows if "wall_s" in r]
        if ok:
            print(json.dumps({"summary": True, "rep": rep + 1, "c": a.concurrency,
                              "median_wall_s": round(statistics.median([r["wall_s"] for r in ok]), 2),
                              "median_ttft_s": round(statistics.median([r["ttft_s"] for r in ok]), 2),
                              "median_decode_tok_s": round(statistics.median([r["decode_tok_s"] for r in ok]), 1),
                              "median_prefill_tok_s": round(statistics.median([r["prefill_tok_s"] for r in ok])),
                              "aggregate_decode_tok_s": round(sum(r["decode_tok_s"] for r in ok), 1)}), flush=True)


if __name__ == "__main__":
    main()
