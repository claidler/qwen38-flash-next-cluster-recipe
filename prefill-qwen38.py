#!/usr/bin/env python3
"""Prefill probe for the qwen-3.8-next lane: cold prefill tok/s at ~8k and ~32k prompts.

Each rep gets a unique nonce prefix so prefix caching does NOT hit (true cold prefill).
Reports usage.prompt_tokens, TTFT, and prefill_tok_s = prompt_tokens / ttft.
A warm repeat of the same prompt shows the prefix-cache hit for comparison.

Usage: prefill-qwen38.py [--reps 3]
"""
import argparse, json, time, urllib.request, uuid

URL = "http://127.0.0.1:8000/v1/chat/completions"
PARA = ("The quick brown fox jumps over the lazy dog while the distributed inference "
        "server routes experts across two GPUs and keeps the KV cache warm. ")


def probe(prompt, max_tokens=4):
    body = {"model": "qwen-3.8-next",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    ttft = None
    usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
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
                if d.get("content") or d.get("reasoning"):
                    if ttft is None:
                        ttft = time.time() - t0
    pt = (usage or {}).get("prompt_tokens", 0)
    return {"prompt_tokens": pt, "ttft_s": round(ttft or 0, 2),
            "prefill_tok_s": round(pt / ttft) if ttft else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    for label, paras in (("8k", 500), ("32k", 2000)):
        for i in range(a.reps):
            row = probe(f"nonce-{uuid.uuid4()}\n" + PARA * paras)
            row.update(rep=i + 1, size=label, mode="cold")
            print(json.dumps(row), flush=True)
        warm_prompt = f"warm-{uuid.uuid4()}\n" + PARA * paras
        probe(warm_prompt)  # populate the cache
        row = probe(warm_prompt + "\nAnswer with one word.")
        row.update(rep=1, size=label, mode="warm")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
