# Two-rank NCCL over a direct CX7 link: the configuration that works and how to tell a wrong leg before the boot hangs

**One sentence: for two ranks on a direct ConnectX-7 cable, list exactly the two HCAs that face the peer
(one per physical port), restrict `NCCL_IB_ADDR_RANGE` to the link's own /24, and keep the patched
ring-only build's full env set — a wrong HCA list passes the preflight and hangs the first collective
with `ibv_modify_qp failed ... curr state INIT, next state RTR`.**

This is the two-rank companion of the four-rank ring configuration in our
[TP4 repository](https://github.com/Keenni/GLM-5.3-Flash-Uncensored-EXL3-4x-DGX-Sparks-TensorFold-TP4)
(which documents the switchless NCCL build and the four-node ring). Read that first for the build; this
page is what changes when the job has exactly two ranks on one cable.

## The env set (per node)

```bash
# dgx03 (rank 0) — both HCAs that face dgx04 on the 10.0.2.0/24 link
NCCL_NET=IB  NCCL_IB_DISABLE=0
NCCL_SOCKET_IFNAME=<mgmt interface>
NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1        # dgx03's two ports toward dgx04 (10.0.2.103/.113)
NCCL_IB_GID_INDEX=<per-node, probed at start>
NCCL_IB_ADDR_RANGE=10.0.2.0/24
NCCL_IB_SUBNET_PREFIX_LEN=24  NCCL_IB_SUBNET_AWARE_ROUTING=1
NCCL_IB_MERGE_NICS=0  NCCL_ALGO=Ring
NCCL_MIN_NCHANNELS=4  NCCL_MAX_NCHANNELS=4
NCCL_SKIP_TREE_CONNECT=1  NCCL_SWITCHLESS_RING_ONLY=1
NCCL_IB_ADDR_FAMILY=AF_INET  NCCL_IB_ROCE_VERSION_NUM=2
NCCL_CROSS_NIC=1  NCCL_CUMEM_ENABLE=0  NCCL_NVLS_ENABLE=0
NCCL_IGNORE_CPU_AFFINITY=1  NCCL_DEBUG=WARN
TORCH_NCCL_ASYNC_ERROR_HANDLING=1
TF_NCCL_LIB=/opt/patched-nccl/libnccl.so.2
```

dgx04 lists *its* two HCAs toward dgx03 (`rocep1s0f0,roceP2p1s0f0`, 10.0.2.104/.114). **The HCA names
are per node and per direction** — the same physical port has a different name on each end of the
cable, and a four-rank ring's list ("the two ports f0/f1 both ways") is wrong for a two-rank job: one
of those ports faces a node that is not in this job, and the collective hangs.

Why exactly two HCAs: with two ports cabled and addressed, two rails double the gather bandwidth; and
the patched build validates a specific listener count per rail — a single HCA can be rejected (fall
back to one HCA only if the log says the two-listener check failed).

## Before you boot: find the right legs

The start script's `gid_of()` probes the first RoCEv2 GID index per node (hard-coding it survives a
reboot only until the GID table reorders). For the legs themselves:

```bash
ibdev2netdev                       # which HCA is which netdev
ip -o -4 addr | grep 10.0.         # which netdev carries the link address
ip route get <peer-link-ip>        # the route must leave via that netdev
ip neigh show dev <netdev>         # ARP entry for the peer = the leg really works
```

ARP is the ground truth: a route that points at the right interface but no neighbor entry means the
peer's IP is not on that cable.

## Failure modes we have hit

| Symptom | Cause | Fix |
|---|---|---|
| First `comm.barrier()` hangs; `NCCL_DEBUG=INFO` shows `ibv_modify_qp failed with 110, curr state INIT, next state RTR` | A listed HCA faces a node not in the job (the ring's list reused) | List only the two HCAs whose ARP entry is the peer |
| `NCCL error 2: unhandled system error` on the first collective | GID index pinned from another boot | Probe `gid_of()` at start; never hard-code |
| The patched build rejects the job | Fewer than the validated listeners per rail | Two addressed HCAs per node (or drop to one if the log insists) |
| Preflight dies with "missing config.json" | The master cannot ssh the worker at all | `ssh-keyscan <both IPs> >> ~/.ssh/known_hosts` — a missing host key *is* an ssh failure, and the script's error message is misleading |
| Everything green but the API is not listening | Rank 0 was started without `--host 0.0.0.0 --port 8000` | The start script adds them on rank 0 only; without them TensorFold serves on 127.0.0.1:8080 |

## Verify before loading weights

A five-second, no-GPU check that the two ranks can talk (adapt the model dir):

```bash
# on each node, in the image:
python3 - <<'EOF'
# tensorfold's own comm: build a 2-rank all_gather, no model, seconds to PASS/FAIL
EOF
```

(In our fleet the script runs this via `scripts/tf_nccl_diag.py` from the TP4 repository; both ranks
must print PASS before the full stack starts.)

## RoCE all-gathers for the hot path

With the link up, `TF_GLM_COMM=roce` routes decode-sized all-gathers (≤256 KiB — the exchange sites
after attention and the MoE block) over b12x's RoCEnante transport in one shot, and keeps large prompt
gathers on NCCL. Measured at TP2: **+4.5% decode prose, +3.8% structured, −19% TTFT, +3–4% prefill**,
bit-exact (the full table in [prefill-decode-tuning.md](prefill-decode-tuning.md)). The engine prints
the split on startup — check for the line before believing the env took effect.
