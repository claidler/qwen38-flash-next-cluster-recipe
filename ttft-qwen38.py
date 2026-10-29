#!/usr/bin/env python3
"""Prefix-cache / prefill test: cold TTFT on an ~8k-token prompt, then warm TTFT
on the same prefix plus a short follow-up. This is the agent-turn workload that
prefix-match-unit + PLE prefill checkpoints target."""
import json, time, urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
PARA = ("The quick brown fox jumps over the lazy dog while the distributed inference "
        "server routes experts across two GPUs and keeps the KV cache warm. ")


def body(prompt, max_tokens=8):
    return {"model": "qwen-3.8-next",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}}


def ttft(prompt):
    req = urllib.request.Request(URL, data=json.dumps(body(prompt)).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                try:
                    o = json.loads(line[5:])
                except json.JSONDecodeError:
                    continue
                for ch in o.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning"):
                        return time.time() - t0
    return time.time() - t0


base = PARA * 500          # ~8k tokens
print(json.dumps({"cold_ttft_s": round(ttft(base), 2)}))
time.sleep(1)
print(json.dumps({"warm_ttft_s": round(ttft(base + "\n\nNow answer with one word: ready?"), 2)}))
