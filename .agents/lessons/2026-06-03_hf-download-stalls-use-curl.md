---
date: 2026-06-03
experiment: memory_prefetch_plan.md Phase 5 (5B/14B model downloads)
category: infrastructure
severity: important
---

# `hf download` stalls indefinitely on rented GPU boxes — use `scripts/curl_hf_fetch.sh`

## What Happened

Downloading `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (~33 GB) and
`Wan-AI/Wan2.1-T2V-14B-Diffusers` (~75 GB) with `hf download` on a fresh
8×H200 box burned ~4 hours and never finished. Throughput was ~1–2 MB/s
in bursts with repeated indefinite stalls (0 bytes for 4+ minutes).
Restarting helped only briefly; a stall-detect-and-restart supervisor
just band-aided the symptom. Worse, the `.incomplete` blob files
sometimes *shrank* across resume attempts, so partial progress was
untrustworthy.

## Root Cause

The Python download path in `huggingface_hub` (0.36.2) was the
bottleneck/failure, not the network: raw `curl` against the same
`https://huggingface.co/.../resolve/...` URLs sustained ~70 MB/s per
stream from the same box. Installing `hf_xet` did not fix it (the
resumed files continued via the HTTP path and stalled again). Exact
client-side trigger not isolated — treat the `hf` CLI as unreliable on
these hosts.

The real mistake was process, not code: stalls were "handled" with
restarts for hours without ever measuring raw network throughput to
rule the network in or out. One 10-second `curl -o /dev/null` test
would have redirected the effort immediately.

## Fix / Workaround

`scripts/curl_hf_fetch.sh <org/repo>` — fetches all missing files of an
HF repo directly with curl (4 in parallel, `-C -` resume, 5 retries)
into the standard HF hub cache layout (`$HF_HOME/hub/models--*`):

- LFS files are written to `blobs/<sha256>` and verified against the
  sha256 that *is* the blob filename (from the tree API `lfs.oid`).
- Non-LFS files are verified with `git hash-object` against their git
  OID (note: the API call in the script filters `lfs == null` files —
  check the final "non-LFS file missing" warnings and fetch those too).
- Snapshot symlinks are created with the correct relative depth, so
  repo-id-based loading (`init_from Wan-AI/...`) works unchanged.

Both Wan models downloaded + verified in ~7 minutes total this way.

## Prevention

- On a new box, before any large HF download: run a 10-second raw-speed
  check (`curl -sL -o /dev/null --max-time 10 -w "%{speed_download}" <resolve-url>`).
  If `hf download` runs far below that, switch to `curl_hf_fetch.sh`
  immediately instead of supervising restarts.
- Generally: when a transfer/job underperforms, measure the underlying
  resource (network, disk, PCIe) FIRST to localize the bottleneck before
  building retry machinery around the broken layer.
