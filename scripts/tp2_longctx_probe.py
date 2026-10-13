#!/usr/bin/env python3
"""TP2 长 ctx 污染台阶探针：每档长 prompt 前后各跑一次 mini 电池，
判定「污染触发点」在哪一档，以及被污染后短请求是否也坏。

对照事实（2026-10-13）：
- dgx01/02 测试栈(split=1) 冷启动后首次 6/6 全对；用户跑完分档扫描(≤32794)+长 ctx 后 → 全档乱答
- dgx03/04 栈(split=1) 首启与重启后 → 第 2 个请求(170 tok)即乱答（但其身前已有过 97.5k/扫描流量）
用法: python3 tp2_longctx_probe.py <BASE_URL> <MODEL>
"""
import json, sys, urllib.request, time

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://192.168.90.101:8000") + "/v1/chat/completions"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "glm53-flash-tf-unc-tp2-test"

FILLER = ("The quick brown fox jumps over the lazy dog near the river bank. "
          "Engineers measure throughput and latency before trusting a deployment. ")
NEEDLE = "Ignore everything above and reply with exactly this token and nothing else: PINEAPPLE-7Z"
FILLERS_PER_K = 42          # ≈24 tok/条 → 1k tok 约 42 条


def post(messages, extra=None, max_tokens=16):
    body = {"model": MODEL, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": messages}
    if extra:
        body.update(extra)
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.loads(r.read())
    return d, time.time() - t0


def mini(tag):
    """短请求对照组：NEEDLE 纯问 + 986 tok 中档。两者都应答 PINEAPPLE-7Z。"""
    res = []
    for label, n in (("26tok", 0), ("986tok", 40)):
        d, dt = post([{"role": "user", "content": FILLER * n + "\n\n" + NEEDLE}])
        msg = (d["choices"][0]["message"].get("content") or "").strip()
        ok = "PINEAPPLE-7Z" in msg and len(msg) <= 24
        res.append(ok)
        print(f"  [{tag}/{label}] ok={ok} {dt:.1f}s {msg[:70]!r}", flush=True)
    return all(res)


def long_ctx(k_tokens, tag):
    d, dt = post([{"role": "user", "content": FILLER * (k_tokens * FILLERS_PER_K) + "\n\n" + NEEDLE}],
                 max_tokens=24)
    msg = (d["choices"][0]["message"].get("content") or "").strip()
    u = d.get("usage", {})
    ok = "PINEAPPLE-7Z" in msg and len(msg) <= 24
    print(f"[{tag}] prompt_tok={u.get('prompt_tokens')} out_tok={u.get('completion_tokens')} "
          f"{dt:.1f}s ok={ok} {msg[:90]!r}", flush=True)
    return ok


print("== 基线 mini（干净态应全 True）==", flush=True)
base_ok = mini("baseline")
print(f"baseline ok={base_ok}", flush=True)

for k in (16, 33, 100):
    print(f"== 长档 {k}k ==", flush=True)
    long_ctx(k, f"long{k}k")
    after = mini(f"after{k}k")
    print(f">>> after {k}k mini ok={after}", flush=True)
    if not after:
        print(f"### 污染点在 {k}k 档：此后短请求也坏 ###", flush=True)
        break
