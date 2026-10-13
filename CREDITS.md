# Credits and provenance

Almost everything that makes this work was built by others. This file follows the chain; corrections
welcome as issues.

## Model

| Piece | Source | License |
|---|---|---|
| GLM-5.3-Flash | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | the model card's license |
| Uncensored FP8 base | [orcarouter/GLM-5.3-Flash-Uncensored-FP8](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-FP8) | as published |
| EXL3 4bpw checkpoint we serve | [neko-legends/GLM-5.3-Flash-Uncensored-EXL3](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3) | ShapleyMcg License 1.0 (attribution required):

> This work includes or was produced using ShapleyMcg, created by Brandon M. Music ([https://github.com/brandonmmusic-max/shapleymcg](https://github.com/brandonmmusic-max/shapleymcg)). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.

## Inference engine and its patches

| Piece | Source | Notes |
|---|---|---|
| TensorFold | [ashhart/TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0) | `NOTICE.md` and `THIRD_PARTY_NOTICES.md` in this repository are his, taken verbatim from the v0.6.6 tag nearest our v0.6.7 base |
| 2× Spark reference stack, container image, 53 patches | [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) | DFlash2 + copy drafts, 4-bit dense weights, FP8 KV, shared pool, vision, tool calling; our image carries her patch set rebased through v0.6.7 (the rebase dossier lives in our [TP4 repository](https://github.com/Keenni/GLM-5.3-Flash-Uncensored-EXL3-4x-DGX-Sparks-TensorFold-TP4)) |
| RoCE all-gather transport | b12x's RoCEnante, shipped inside TensorFold (`TF_GLM_COMM=roce`) | we configured and measured it, we did not write it |
| DFlash2 drafter | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) | CC BY-NC-ND 4.0 — non-commercial use only |
| Switchless NCCL 2.30.7 | [alexellis/switchless-nccl](https://github.com/alexellis/switchless-nccl) (Alex Ellis, OpenFaaS Ltd) on [SparkRing](https://github.com/FujitsuPolycom/sparkring) patches; prior art [josephdrose/nccl-spark-switchless](https://github.com/josephdrose/nccl-spark-switchless) | the NCCL patch is not ours; our contribution is the per-topology env set and the diagnostics |
| TP4 reference & sequence-parallel design | [TensorFold PR #159](https://github.com/ashhart/TensorFold/pull/159) by drowzeys | a separate `glm_moe_dsa` family (753B) — not portable as code to `glm5_next`; notably his indexer top-k is *adapted from* this engine's `glm5_next` `_select_rows` |

## Ours (Apache-2.0)

- The world=2 hyper-connection row-split root cause and fix
  ([docs/hcsplit-w2-rca.md](docs/hcsplit-w2-rca.md)) — one branch of `hcsplit.py`'s `partial()`,
  restored to the upstream v0.6.0 semantics.
- The per-step latency instrumentation (the `stages=` log field, the select proj/scan split, the
  multi-profile startup line) — small additions to `server.py`/`forward.py`.
- The tuning measurements and their conclusions
  ([docs/prefill-decode-tuning.md](docs/prefill-decode-tuning.md)) — the kept changes are environment
  variables; the rejected ones are documented so nobody re-tests them.
- The two-rank NCCL configuration for a direct CX7 link
  ([docs/nccl-2rank-direct.md](docs/nccl-2rank-direct.md)) and the start/stop scripts.
- The acceptance battery and probes (`tp2_battery.py`, `tp2_longctx_probe.py`, `tp2_decode_probe.py`).

## Runtime stack

NVIDIA's PyTorch container (redistributed as a value-added runtime image in the lineage of Mia's
kit); its notice is printed by the container at every start.

## Development

Developed with the [Hermes](https://hermes-agent.nousresearch.com) agent; cross-checked daily against
our 4× Spark TP4 line ([companion repository](https://github.com/Keenni/GLM-5.3-Flash-Uncensored-EXL3-4x-DGX-Sparks-TensorFold-TP4)).

## Corrections

Spotted a mis-attribution? Open an issue.
