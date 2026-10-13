# The world=2 hyper-connection row-split: deterministic wrong answers, root cause, fix

**One sentence: the split's non-overlap path at world=2 never filled the rank's own rows of the fp32
partial buffer, so the glue added stale garbage into every prompt chunk of ≥64 rows; the fix is one
line, and the bug is invisible to every check we had.**

This is the postmortem of a two-day hunt on our 2× DGX Spark line (TensorFold v0.6.7, `glm5_next`
family). It also settles *why TP4 was correct while TP2 was wrong on the same image* — the two run
different branches of the same function, and only one of them had been repaired.

## Symptom

With `TF_GLM_HC_SPLIT=1` (and either exchange — `p2p` or `gather`):

- Prompts under `TF_GLM_HC_SPLIT_MIN_ROWS` (64 rows) answer correctly, bit-exact with the unsplit
  engine (the smoke test's sha matches the TP4 baseline).
- Prompts of ≥64 rows answer **deterministically wrong**: the same prompt repeated gives the identical
  wrong answer (not a race); a needle instruction ("reply with exactly PINEAPPLE-7Z") is ignored, the
  model echoes prompt fragments or emits noise. A 986-token prompt produced
  `sha=3cdd8c61b882` where the unsplit engine gives `0a49433aa27b`.
- `"draft": false` and drafted replies disagree — bit-exactness broken across the board.
- The wrong sha **differs between machine pairs** (dgx01/02: `3cdd8c61b882`; dgx03/04: `36a4666e119d`)
  and is stable within a pair — stale buffer contents, not logic alone.
- Long-context behavior ("after a >100k prefill the engine answers wrong") is the *same* bug: a long
  prompt necessarily enters the split path; a short probe afterwards is fine (a 16k prompt does not
  pollute later short requests — we checked).

## Why every check passed

- **The startup smoke test uses a 10-token prompt** — under the 64-row gate, so it runs unsplit and
  cannot see the bug. Two boots "verified" this way were wrong on every longer prompt.
- **The exchange itself was verified bit-exact** — in-process comparisons of the all-to-all assembly
  against `all_gather` showed 0 mismatched elements. The bug is *before* the exchange.
- **It reproduced on both machine pairs** after an earlier "not reproducible" reading — which turned
  out to be a probe bug (a string replace that silently didn't apply, so the requests hit the
  production stack instead of the debug stack; the production engine's log had no matching `done req`
  lines). Always verify the probe's BASE/MODEL by reading the file or matching engine logs.

## Root cause

`families/glm5_next/cuda/hcsplit.py`, `HcSplit.partial()`, the non-overlap path (`stream is None`,
i.e. `TF_GLM_PREFILL_OVERLAP=0` — our default). After the world=4 repairs, the branches read:

```python
if self.stream is None:                        # no second stream: fill and swap straight through
    if self.world > 2:
        self._fill(fill, 0, self.R)            # every rank needs its partials for every rank's rows
        self._swap_partial(0)
        return
    for j, (off, n) in enumerate(self.pieces): # over two ranks:  ← THE BUG
        self._fill(fill, theirs + off, theirs + off + n)   # only the PEER's half is filled
        self._swap_partial(j)
    return
```

`fill(lo, hi)` computes this rank's fp32 partial rows into `b.part[lo:hi]` (a row-independent prompt
projection — `out_proj`'s callback is correct for any range). The glue then reads **both** halves:

```python
own = b.part[lo:hi]                    # lo..hi are THIS RANK's rows
g0, g1 = (own, got) if self.rank == 0 else (got, own)
glue.hc_post_pair(x, x, g0, g1, ...)
```

`got[j]` holds the peer's partial of this rank's rows (received via the swap); `own` is supposed to be
this rank's partial of its own rows — **but the loop only ever fills `theirs + off`**. Rows
`[mine, mine + H)` of `b.part` are never written for this chunk: the glue adds whatever the buffer held
from an earlier chunk (or zeros). Deterministic, machine-pair-dependent, invisible to exchange-level
comparisons.

**Why TP4 was fine on the same image**: the `world > 2` branch was *repaired* during the TP4 work
(its own earlier bug — an inherited "fill only the peer's block then return" — was deleted and replaced
with `fill(0, R)` + a private scratch buffer). The world==2 non-overlap branch kept the disease.
Mia's original v0.6.0 `hcsplit.py` (two ranks only) fills the whole row range:

```python
if self.stream is None:
    fill(0, self.R)
    for j in range(len(self.pieces)):
        self._swap_partial(j)
    return
```

The world=2 *overlap* path in our tree is also correct — it fills own rows explicitly at the end
(`self._fill(fill, mine, mine + self.H)`). Only the non-overlap W2 branch lost it, which is why the
bug needed `PREFILL_OVERLAP=0` to show.

Ruled out along the way: `b.gath` aliasing (the W4 hazard — at world 2 the `got` slices and the gather
layout agree, and nothing else writes `gath` between `partial()` and `glue()`); the `gather` vs `p2p`
exchange choice (both wrong, same shape); startup count/nondeterminism (deterministic per machine).

## The fix

Restore the upstream semantics in the W2 non-overlap branch:

```python
    self._fill(fill, 0, self.R)                # over two ranks too: the glue reads b.part of this rank's
    for j, (off, n) in enumerate(self.pieces): # own rows (Mia's original), not just the peer's half
        self._swap_partial(j)
```

One line of behavior (plus a comment). **Verification** (battery `scripts/tp2_battery.py` + step probe
`tp2_longctx_probe.py`):

- 6/6 battery: control / short (170 tok) / mid (986) / long (3866) / repeat / serial — all correct.
- **Every tier's reply sha equals the unsplit baseline `0a49433aa27b`** (was `3cdd8c61b882` at 986).
- The 10-token smoke sha unchanged (`694aab7656bc`).
- 16k / 33k / 100k prompts correct, and the mini-battery after each is clean — no pollution.

## The verdict on HC_SPLIT at world 2

Measure the fix honestly and the split buys **nothing at TP2**: with the repaired code, cold prefill is
1696/1659/1603 tok/s (16k/33k/100k) with the split and ~1600–1700 without — same band. Two reasons:

1. **The gain's ceiling is small.** Row-splitting divides only the glue kernels (hc_post/hc_pre/stream
   means) — ~10% of a prompt chunk's time. The fp32 partial projections (`fill`) must be computed for
   *all* rows on every rank either way: each rank needs its own rows' partials (the glue reads them)
   *and* the peer's rows' partials (to send them). The fill is the expensive part (24% in the MoE block
   alone), and splitting does not touch it.
2. **The ~1900 tok/s we once measured with the split was the bug's shadow**: the broken branch filled
   only the peer's half — half the fill work, wrong answers. Speed by skipping work.

At world 4 the picture differs (bigger gathers, more ranks to sum) and the split measured +16% — that
line has its own repository. `PREFILL_OVERLAP=2` (hiding the exchange on a second stream) measured 1646/
1642/1630 tok/s: no gain either — one GPU, two streams, the same SMs; and the p2p exchange on a 200G
direct link is short anyway.

**Production runs `TF_GLM_HC_SPLIT=0`.** The image carries the fixed file either way; the start script's
comment says so.

## ⚠️ Traps

- **Smoke tests prove nothing about the split** — 10 tokens < 64 rows. Any change to the split path
  needs the length-tiered battery. This bit us twice on the TP4 line and once here.
- **A probe that edits its own BASE/MODEL by string replace must be re-read or matched against engine
  logs** — our first "not reproducible on dgx01/02" reading was requests silently landing on another
  stack.
- **The policy string for noise-aware rules needs the literal `noisy`** (`fcost7:noisy:0.3`): a bare
  `fcost7:0.3` makes both ranks die in `encode_policy` at startup, and the error line sits in the rank
  log, not the start script's output. Validate a policy inside the container before restarting on it.
- In-process comparisons of a new exchange against `all_gather` must allocate the reference as
  `world × send` **separately** — reusing the engine's buffer gives a size error that looks like an
  exchange bug.
