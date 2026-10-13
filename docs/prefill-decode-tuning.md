# Prefill and decode, step by step: what each part costs and what tuning buys

**One sentence: at TP2 the big prefill blocks are the MoE expert kernel (24%) and the DSA indexer scan
(15%), decode is dominated linearly by the verify window (+4.7 ms per drafted row), the only wins are
environment variables (RoCE all-gathers, MSE draft quantization, a score row-block bump — together
+6% decode prose, +5% prefill), and the "obvious" candidates — a radix top-k port, one-pass attention,
depth policies — are already in the tree or measurably worse.**

All numbers are ours, on 2× DGX Spark (GB10), TensorFold v0.6.7, warm kernel caches, temperature 0,
thinking off, fresh (uncached) prompt tokens. Every configuration below is verified bit-exact against
the untuned engine: a fixed battery's reply sha256 is identical across all of them (`0a49433aa27b` for
the prompt tiers, `478cc5a22344`/`7451ea6fe928` for the decode probes, `694aab7656bc` for the 10-token
smoke). Correctness first, then speed.

## Measuring per step (the instrumentation)

Three channels, all shipped in our image, all zero-cost to keep on:

1. **The `done req` log line** now carries `stages={...}` on serial requests: draft / forward / sample /
   commit, ms per request. (Under `--parallel > 1` requests take the multi-stream path, which does not
   produce it; use channel 3.)
2. **The prompt-chunk profile** (`TF_GLM_PROFILE=1`): one line per chunk, every block timed with device
   sync — `moe: routed (exl3)`, `dsa: select proj/scan`, `dsa: sparse attention`, `kda: ...`, `hc`, ...
   We also split the indexer's projection from its scan (the `select proj` / `select scan` lines).
3. **`TF_GLM_MULTI_PROFILE=1`** (startup, needs both ranks): prints the verify-window cost per drafted
   row (one-stream graphs vs batched graphs) and one round's wall-clock split (plan / message / propose /
   verify / sample / accept / commit / taps / emit).

## Decode, step by step

From `TF_GLM_MULTI_PROFILE=1` on a lone stream:

| Rows in the verify window | One-stream graph | Batched graph |
|---|---|---|
| 1 | 27.35 ms | 27.63 ms |
| 2 | 36.00 | 36.38 |
| 4 | 50.39 | 51.35 |
| 8 | 69.93 | 70.68 |
| 16 | 97.30 | 97.96 |

- **The batched graphs cost ≤2%** over the one-stream graphs at every width — multi-stream decode is
  not leaving anything on the table; contention is physics, not misconfiguration.
- **Cost is linear in drafted rows: ~4.7 ms per row on top of a ~27 ms base.** A round with k drafts is
  worth it only if the expected accepted tokens per (base + 4.7·k) ms beats a k−1 round. That is the
  entire decode optimization problem on this engine, and it is why depth knobs below disappoint.
- A round's other parts are small: propose 3.7 ms (the DFlash2 drafter), message/plan/sample/commit
  ~2–3 ms. The arithmetic closes: prose at 104 rounds × (44.5 ms verify + 3.7) ≈ the observed
  47–48 ms/round, i.e. the measured 52 tok/s.

**Depth-policy A/B (the rejected one first):**

| Policy | Rounds | Drafted/accepted | prose tok/s | struct tok/s |
|---|---|---|---|---|
| `fnc7:0.3` (production: noise-aware confidences, β=2.0) | 104 | 151/237 | **52.0** | **83.1** |
| `fcost7:noisy:0.3` (cost-optimal depth from the startup table) | 93 | 162/603 | 38.2 | 77.0 |
| `fnc7:0.3:1.0` (β=1.0, chains run longer) | 104 | 151/237 → 164/362* | 49.5 | 79.7 |

*β=1.0 drafts more tokens per round (tpr 2.45 → 2.80) but each round verifies linearly more rows —
net −5%. The depth/acceptance trade is already at the optimum; both experiments agree. Note the
`fcost` trap: the literal `noisy` is required, a bare `fcost7:0.3` kills both ranks in `encode_policy`.

**What did work (kept):**

| Change | decode prose | decode structured | TTFT |
|---|---|---|---|
| baseline (nccl all-gathers, minmax draft quant) | 49.0 | 79.7 | 0.32 s |
| + `TF_GLM_COMM=roce` (≤256 KiB all-gathers on RoCE, large on NCCL) | 51.2 (+4.5%) | 82.7 (+3.8%) | 0.26 s |
| + `TF_GLM_DRAFT_QUANT=mse` (least-squares draft copies; drafts only propose) | **52.0 (+6.1%)** | **83.1 (+4.3%)** | **0.25 s** |

`roce` is b12x's RoCEnante transport shipped in TensorFold: the decode-sized all-gathers go over the
direct CX7 link in one shot (NCCL's ring would add latency), large prompt gathers stay on NCCL. Logs
confirm: `all-gathers up to 256 KiB over RoCE (rocep1s0f0, rocep1s0f1...), larger ones over NCCL`.
`mse` quantizes the drafter's 4-bit copies by least squares instead of min/max — the drafter only
*proposes*, never verifies, so replies cannot change (the docstring says so; the sha proves it).

## Prefill, step by step

One 8,192-row prompt chunk, total 5.0–5.1 s (tuned config):

| Block | Time | Share |
|---|---|---|
| moe: routed (exl3) | 1.20 s | 24% |
| dsa: select scan (indexer scoring + top-k) | 0.73 s | 15% |
| kda: projections | 0.46 s | 9% |
| dsa: sparse attention | 0.41 s | 8% |
| kda: out + all-gather | 0.31 s | 6% |
| hc + hc (layer end) | 0.50 s | 10% |
| moe: combine / shared expert / route | 0.39 s | 8% |
| everything else | ~0.1 s | 2% |

The indexer's projection (`dsa: select proj`) is 0.02 s — negligible; the scan is the cost. The scan is
**already a single Triton kernel** (a radix select over float scores: 4×8-bit histogram passes, then one
ordered pass — `sparse.py:_select_rows`). We checked drowzeys' `glm_moe_dsa/cuda/topk.py` line by line:
it is the *same algorithm, adapted from this engine's `_select_rows`* (his docstring says so). **There is
nothing to port.** What remains in "select tokens" is the indexer's scoring itself — O(R×NP×H×D), the
DSA indexer's intrinsic work.

One-pass sparse attention is also already on (`TF_GLM_SPARSE_ONEPASS=1`): one program per row walks the
selected tokens in a single online softmax, no partials/merge. (Turning it off and merging 512-token
chunks is the slower path.)

**What worked (kept):** `TF_GLM_SCORE_RB=8` (rows per scoring program, default 4) — the scan drops
0.81 → 0.73 s (−10%), worth ~2% of prefill end to end. 8 is the maximum the kernel accepts.

**Measured as worthless (do not re-test):** `TF_GLM_HC_SPLIT=1` with the fix (1696/1659/1603 vs the
same band unsplit — see the RCA document for why its ceiling is the glue's ~10%), and
`TF_GLM_PREFILL_OVERLAP=2` (1646/1642/1630: one GPU, two streams, same SMs).

| Config | prefill 16k | 33k | 100k |
|---|---|---|---|
| baseline | 1696 | 1659 | 1603 |
| tuned (roce + mse + RB8) | **1777** | **1725** | **1687** |

## What is left (kernel-level, not env-level)

- The MoE expert kernel's tile schedule (24% of a chunk) — real work, needs its own campaign.
- The verify window's ~4.7 ms per drafted row — the decode ceiling; shrinking it changes every depth
  trade-off.
- The indexer's scoring arithmetic (15%) — intrinsic to DSA at this index geometry.

## ⚠️ Traps

- **`TF_GLM_CACHE_GIB` is a cap, not an allocation** — set it large (we use 1000) and let the engine
  size the pool from `MemAvailable − RESERVE`.
- **`TF_GLM_DFLASH_POLICY` strings must be validated in-container before a restart** (`fnc7:0.3:1.0`
  parses; `fcost7:0.3` does not — the literal `noisy` is missing).
- **Measure on fresh tokens**: `usage.prompt_tokens_details.cached_tokens` or the engine's `cached=`
  field; a 100k prompt that resumes from a kept state prefills in 0.2 s and tells you nothing about
  prefill speed.
- **Concurrency probes on a production engine eat its cache** — a probe request is a new allocation
  that can evict a warm session; use the debug pair.
