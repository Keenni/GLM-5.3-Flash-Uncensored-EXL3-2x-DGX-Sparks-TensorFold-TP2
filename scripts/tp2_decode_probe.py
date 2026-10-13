#!/usr/bin/env python3
"""TP2 decode A/B 探针：固定 prompt + max_tokens，读引擎日志 tok/s 对比 draft 策略。

用法: python3 tp2_decode_probe.py <BASE> <MODEL> <tag>
每个用例跑 3 次，打印引擎 done req 的 tok/s（从日志抓，最后统一汇总）。
"""
import json, sys, urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://192.168.90.101:8000") + "/v1/chat/completions"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "glm53-flash-tf-unc-tp2-test"

FILLER = ("The quick brown fox jumps over the lazy dog near the river bank. "
          "Engineers measure throughput and latency before trusting a deployment. ")

CASES = [
    ("prose", FILLER * 40 + "\n\nWrite a detailed 300-word essay about the history of computing."),
    ("struct", FILLER * 40 + "\n\nOutput a JSON array of 30 objects, each with fields name, value, tag. "
              "No other text, no markdown fences."),
]


def post(prompt, max_tokens=256):
    body = {"model": MODEL, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"].get("content") or ""


if __name__ == "__main__":
    for label, prompt in CASES:
        for i in range(3):
            msg = post(prompt)
            print(f"[{label}#{i}] out_chars={len(msg)} done", flush=True)
