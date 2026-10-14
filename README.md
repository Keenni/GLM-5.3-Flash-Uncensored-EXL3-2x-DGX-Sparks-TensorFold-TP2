<h1 align="center">GLM-5.3-Flash Uncensored EXL3 on 2× DGX Spark — TensorFold TP2, unified v0.6.7</h1>

<p align="center">
  <sub>Two-rank TensorFold on a direct ConnectX-7 link: the v0.6.7 unified image, a root-caused and
  bit-exact row-split fix, RoCE all-gathers for decode, and per-step latency instrumentation.
  Companion to our <a href="https://github.com/Keenni/GLM-5.3-Flash-Uncensored-EXL3-4x-DGX-Sparks-TensorFold-TP4">4× Spark TP4 repository</a>.</sub>
</p>

---

Serve [neko-legends/GLM-5.3-Flash-Uncensored-EXL3](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3)
(EXL3 TR3 4bpw, ~176 GB, uncensored/abliterated) from **two** NVIDIA DGX Sparks (GB10, 128 GB unified memory each)
as one tensor-parallel engine through an OpenAI-compatible API, with the model's full 1,048,576-token context,
DFlash2 speculative decoding, tool calling, image and video input, and `/metrics`.

The reference two-rank deployment of this engine is
[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
(v0.5.0 + 52 patches). This repository is what we run on top of a **v0.6.7 unified image** (one image serving
both our TP2 and TP4 lines, all patches baked in — no bind-mount overlays):

- **A root-caused, bit-exact fix of the hyper-connection row-split** (`TF_GLM_HC_SPLIT`) at world=2 —
  the split's W2 non-overlap path never filled the rank's own rows of the fp32 partial buffer, so every
  prompt of ≥64 rows answered deterministically wrong. Full postmortem:
  [`docs/hcsplit-w2-rca.md`](docs/hcsplit-w2-rca.md).
- **Measured, kept and rejected tuning** for prefill and decode, with the engine's per-step latency
  exposed (verify windows by row, per-round propose/verify/commit, the MoE / indexer split of a prompt
  chunk): [`docs/prefill-decode-tuning.md`](docs/prefill-decode-tuning.md).
- **A working 2-rank NCCL configuration over a direct CX7 link** (two HCAs per node, the patched
  ring-only build, the exact env set and the checks that find a wrong leg before the boot hangs):
  [`docs/nccl-2rank-direct.md`](docs/nccl-2rank-direct.md).
- **The spill tier never wrote a byte in `--parallel` mode** — upstream #423 wired disk spill and
  disk-resume into the single-stream engine only; the multi-stream scheduler's evictions silently
  discarded kept prompts (0 bytes written in 30 h while 42k–220k-token conversations ate 27–211 s full
  re-prefills). Wired in two overlay files, both stacks: [`docs/spill-parallel-wiring.md`](docs/spill-parallel-wiring.md).

## Credits and provenance

Almost everything that makes this work was built by others. Please read [CREDITS.md](CREDITS.md); the
short version:

| Piece | Source |
|---|---|
| Model GLM-5.3-Flash | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) by Z.ai |
| Uncensored FP8 base of the checkpoint | [orcarouter/GLM-5.3-Flash-Uncensored-FP8](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-FP8) |
| EXL3 4bpw checkpoint we serve | [neko-legends/GLM-5.3-Flash-Uncensored-EXL3](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3) (ShapleyMcg License 1.0, attribution required — see [CREDITS.md](CREDITS.md)) |
| Inference engine | [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart (Apache-2.0; `NOTICE.md` and `THIRD_PARTY_NOTICES.md` are his, taken from the v0.6.6 tag nearest our base) |
| 2× Spark reference stack, container image, 53 patches | [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) |
| Switchless NCCL 2.30.7 build | [alexellis/switchless-nccl](https://github.com/alexellis/switchless-nccl) (Alex Ellis, OpenFaaS Ltd) on SparkRing patches; prior art [josephdrose/nccl-spark-switchless](https://github.com/josephdrose/nccl-spark-switchless) |
| DFlash2 drafter | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (CC BY-NC-ND 4.0 — non-commercial) |
| RoCE all-gather transport | b12x's RoCEnante, shipped inside TensorFold (`TF_GLM_COMM=roce`) |
| TP4 reference & sequence-parallel design | [TensorFold PR #159](https://github.com/ashhart/TensorFold/pull/159) by drowzeys (a separate `glm_moe_dsa` family for the 753B model; its indexer top-k is adapted *from* this engine's `glm5_next` `_select_rows`) |
| Development | The stack was developed with the [Hermes](https://hermes-agent.nousresearch.com) agent, and cross-checked against our 4× Spark TP4 line (see companion repository) |

The start/stop scripts, the tuning measurements, the row-split fix and its documentation are ours
(Apache-2.0, see [LICENSE](LICENSE)). We are not affiliated with any of the parties above; bugs in the
row-split fix, the scripts and the tuning conclusions are ours, not theirs.

## Results

Measured on our fleet (2× DGX Spark on a direct CX7 link, warm kernel caches; self-measured, orientation
not leaderboard — Mia's kit measures with sparkDash and capped clocks, so numbers are not directly
comparable):

| Item | Value |
|---|---|
| KV pool at start | **1,048,592 tokens** shared across streams (single-request cap ~1.048M; `--parallel 12`) |
| Ready | 123–142 s with warm kernel caches (first boot compiles) |
| Cold prefill, 16k / 33k / 100k prompts | **1,777 / 1,725 / 1,687 tok/s** (fresh tokens, tuned: RoCE + `DRAFT_QUANT=mse` + `SCORE_RB=8`) |
| Decode, single stream | **52.0 tok/s prose, 83.1 tok/s structured**, TTFT 0.25 s (from 49.0 / 79.7 / 0.32 s before the tuning — +6.1% / +4.3% / −22%) |
| Draft acceptance | 5/6 on our battery, 63.7% of drafted tokens on prose |

Every change behind those numbers is one environment variable or one file, verified bit-exact against the
untuned engine (the reply sha256 of a fixed battery is identical across all configurations — see
[`docs/prefill-decode-tuning.md`](docs/prefill-decode-tuning.md) for the A/B table, including the
**rejected** variants: the `fcost` depth policy −22%, β=1.0 noise-aware confidence −5%, `HC_SPLIT` at
world 2 worth ~0%, `PREFILL_OVERLAP=2` worth ~0%).

Known shape of the stack, so nobody is surprised:

- **The row-split fix is in the image** (it also serves the TP4 line), but our TP2 production runs
  `TF_GLM_HC_SPLIT=0`: at world 2 the split buys ~0% (its ceiling is the glue's ~10% share, and the
  broken variant that showed ~1900 tok/s was skipping half the fill work — see the RCA document).
- **Prefix cache hits require a stable, shared system prompt**; no system message → 0% by design.
- The checkpoint is uncensored/abliterated; behavior is the checkpoint's, not the engine's.

## Repository layout

```
scripts/
  start-glm53-flash-tf-unc-tp2-v067.sh   the production start (master-first, built-in preflight /
                                         memory gate / smoke test; run ONCE on the master node)
  stop-glm53-flash-tf-unc-tp2-v067.sh    master-first stop
  start-glm53-flash-tf-unc-tp2-test.sh   an identical script for a second pair of nodes (debugging:
                                         keep a broken configuration reproducible while production
                                         serves)
  stop-glm53-flash-tf-unc-tp2-test.sh
  tp2_battery.py                         the acceptance battery — smoke tests CANNOT see the row-split
                                         bug (10 tokens < 64 rows); this can. Length tiers + determinism
                                         + drafted/serial bit-exact
  tp2_longctx_probe.py                   the step probe: 16k/33k/100k prompts, mini-battery after each,
                                         catches pollution and gives clean prefill rates
  tp2_decode_probe.py                    fixed decode prompts (prose / structured) for tok/s A/Bs
docs/
  hcsplit-w2-rca.md                      the row-split bug: symptom, root cause with the code, why
                                         every check passed, the fix, the verdict on HC_SPLIT at TP2
  prefill-decode-tuning.md               per-step latency, the full A/B table (kept and rejected),
                                         what is already optimal in the tree (do not re-test)
  nccl-2rank-direct.md                   the 2-rank NCCL/RoCE configuration and its failure modes
```

## Deployment sketch

Read the header of `scripts/start-glm53-flash-tf-unc-tp2-v067.sh` first — it contains the memory
arithmetic (weights ~82 GiB per rank, the pool formula, the `RESERVE` trade), the dependencies, and the
seven deltas from our TP4 script. The short version:

1. Two DGX Sparks cabled CX7-to-CX7, IPv4 on both ends of the link, `ping` working.
2. On both nodes: the `tensorfold-glm53` image (see the TP4 repository's build notes for the v0.6.2 →
   v0.6.7 lineage), the checkpoint and drafter under `/models`, the patched `libnccl.so.2`.
3. Key-based ssh from master to worker; `ssh-keyscan` both nodes first (a missing host key fails the
   preflight with a misleading error).
4. `bash start-glm53-flash-tf-unc-tp2-v067.sh` **once, on the master** — it starts rank 1, then rank 0,
   waits, and runs a smoke test. It refuses to touch any other container and dies politely if :8000 is busy.
5. **Run the battery, not just the smoke test** (`tp2_battery.py`): the smoke test's 10-token prompt
   cannot reach the row-split path, and two of our "all green" boots were wrong on every prompt ≥64 rows.
