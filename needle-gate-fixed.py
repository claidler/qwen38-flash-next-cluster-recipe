#!/usr/bin/env python3
"""Corrected needle gate for the Qwen3.8-Flash-Next lane.

Fixes the 29/30 Aug 2026 needle-gate confound: the original needle-gate.py sent
max_tokens=64 with thinking-on, so on a hard recall task the model burned the
whole budget on reasoning and returned EMPTY content -> FAIL with garbage=no.
Empty-content FAILs are gate artifacts, not recall failures. This variant gives
the model a real answer budget (1024) and low reasoning effort, so a FAIL now
means the model actually could not recall the needle.

Also note: hex-filler haystacks tokenize ~1.8 tokens per nominal line-token, so
"depth 120k" serves ~215k real prompt tokens — quote usage.prompt_tokens, not
the nominal depth, when reporting.

Usage: python3 needle-gate-fixed.py [depth_k] [reps]
"""
import json, os, random, sys, time, urllib.request

API = os.environ.get("AB_API", "http://192.168.86.179:8888")
MODEL = os.environ.get("AB_MODEL", "Qwen3.8-Flash-Next")
DEPTH_K = int(sys.argv[1]) if len(sys.argv) > 1 else 120
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
API_URL = API + "/v1/chat/completions"

rng = random.Random(42)

def haystack(n_tokens):
    lines = []
    pool = "abcdef0123456789"
    while len(lines) * 40 < n_tokens:
        lines.append("".join(rng.choice(pool) for _ in range(80)))
    return "\n".join(lines)

def run(depth_k):
    needle = f"NEEDLE-{rng.randint(100000, 999999)}-{rng.randint(100000, 999999)}"
    hs = haystack(depth_k * 1000)
    lines = hs.split("\n")
    idx = int(len(lines) * 0.75)
    lines[idx] = needle
    hs = "\n".join(lines)
    prompt = (
        "Below is a large text. Find the line that looks like a marker "
        "starting with NEEDLE- and reply with ONLY that line, nothing else.\n\n"
        f"---TEXT START---\n{hs}\n---TEXT END---\n"
    )
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.0,
        "reasoning_effort": "low",
        "stream": False,
    }
    req = urllib.request.Request(API_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    dt = time.time() - t0
    content = out["choices"][0]["message"].get("content") or ""
    reasoning = out["choices"][0]["message"].get("reasoning") or ""
    if not content:
        content = out["choices"][0]["message"].get("reasoning_content") or ""
    ok = needle in content
    garbage = "!!!!" in content or "!!!!" in reasoning
    n_in = out.get("usage", {}).get("prompt_tokens", 0)
    n_out = out.get("usage", {}).get("completion_tokens", 0)
    return ok, garbage, content.strip().replace("\n", " ")[:120], n_in, n_out, dt

def main():
    print(f"needle-gate-fixed: depth={DEPTH_K}k reps={REPS} effort=low max_tokens=1024 api={API} model={MODEL}")
    results = []
    for i in range(REPS):
        try:
            ok, garbage, snippet, n_in, n_out, dt = run(DEPTH_K)
            results.append(ok)
            print(f"  rep{i+1}: {'PASS' if ok else 'FAIL'} garbage={'YES!!' if garbage else 'no'} "
                  f"in={n_in} out={n_out} tok {dt:.0f}s :: {snippet}")
        except Exception as e:
            results.append(False)
            print(f"  rep{i+1}: ERROR {e}")
    print(f"VERDICT: {sum(results)}/{REPS} passed at {DEPTH_K}k")
    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())
