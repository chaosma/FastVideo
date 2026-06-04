# Activation Streaming (STREAMED_OFFLOAD) — Phase 5 & 6 Validation Report

Full findings from the GPU validation of the activation-streaming work
(`memory_prefetch_plan.md`, Phases 0–6) on an 8× H200 SXM box, 2026-06-03/04.

**TL;DR.** Streamed activation offload reduces the Wan2.1-14B 4K backward
peak from **95.3 GB → 71.1 GB per card (−25 %) at zero step-time cost
(1.0005× FULL)**, stable over a 10-step soak. The plan's < 60 GB target is
**missed by ~11 GB**; the gap is fully attributed (Torch-SDPA backward
transient + ~11 GB of non-block saved tensors) with concrete follow-ups.
Phase 5 (5B) passed 9/9 after fixing four real bugs the harness exposed in
the Phase 3 prefetch scheduler.

---

## 1. Hardware / software

- 8× NVIDIA H200 SXM (143,771 MiB ≈ 140 GB each), 2 TB host RAM.
- Python 3.11.15, torch 2.11.0+cu128, venv `/workspace/venv/main`.
- **Attention backend: Torch SDPA** — `flash_attn` is not installed in this
  venv. This matters for the Phase 6 gap analysis (§5).
- Branch `peak-mem-reduction` @ `b81a0f6` ("Phases 3 & 4") + the fixes in §3.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (run.sh default since
  Phases 3 & 4 commit) on all runs.

## 2. What was validated

| Mode | Wrapping | Saved block inputs live… |
|---|---|---|
| FULL (baseline) | `checkpoint_wrapper(block)` | on GPU, all 30/40 layers |
| FULL_OFFLOAD | `offload_wrapper(checkpoint_wrapper(block))` | pinned CPU, sync copies on compute stream |
| STREAMED_OFFLOAD | `StreamedOffloadCheckpointWrapper(checkpoint_wrapper(block))` | pinned CPU, async copies on a dedicated stream + one-layer-ahead backward prefetch |

All modes recompute identically (full per-block); they differ only in where
the saved block inputs sit between forward and backward.

Harness: `scripts/4k_milestone/phase5_validation.sh` (pytest suite → three
2-step probed runs → `phase5_verify.py` grading) and
`scripts/4k_milestone/phase6_14b_study.sh` (same harness at the 14B
measurement point + 10-step streamed soak).

## 3. Phase 5 — 5B / 4×H200 / 704×1280 × 121f

### 3.1 First run: FAIL — four real bugs found and fixed

The initial harness run failed 4/18 unit tests. All four were diagnosed and
fixed in `fastvideo/training/activation_streaming.py` (+ one test):

1. **Cross-model registry contamination.** `next_bwd_block()` consulted
   `fwd_order`, a first-seen-ever list that accumulates block indices across
   every wrapper set built in a process (the registry and the block-idx
   counter are process-global). The scheduler would prefetch blocks of dead
   models, leaking `{idx: []}` live-buffer entries and inflating the
   observed prefetch queue to 3 layers. **Fix:** derive backward order from
   the current step's `records` insertion order (re-inserted by `new_call`
   each forward, popped after backward) — exactly the pending-backward set.
2. **Early-firing `full_backward_hook`.** For a module none of whose inputs
   requires grad (a model's first block fed a non-requires-grad leaf),
   PyTorch fires the full backward hook at *output*-grad time — before the
   block's own backward/unpack has run (PyTorch warns about exactly this).
   `release_after_bwd` then destroyed records the imminent unpack needed →
   `KeyError` on step 2. **Fix:** skip cleanup while a block's live slots
   are unconsumed; the finalizer (below) sweeps it instead.
3. **No guaranteed end-of-backward cleanup.** Registry hygiene depended on
   fortunate hook ordering. **Fix:** `arm_finalizer()` registers a
   once-per-backward `torch.autograd` `queue_callback` that clears all
   remaining registry state before `loss.backward()` returns — making
   "registry is clean after every step" an invariant.
4. **Over-strict bit-exact test.** `test_wrapper_forward_backward_matches_unwrapped`
   asserted `atol=0` on grads. The unpacked buffer lives at a different
   address/alignment than the original activation, so cuBLAS may select a
   different reduction kernel; measured deviation is 1–2 fp32 ULPs (~1e-7),
   forward bit-exact. **Fix:** relax grads to `atol=1e-6/rtol=1e-5`
   (forward still asserted bit-exact), as §10 of the plan anticipated.

None of bugs 1–3 affected the integration probes (one model per process;
DiT block inputs require grad) — which is why training runs passed even
before the fixes. The harness's verifier also had a log-format bug (loss
regex matched `loss=`, trainer emits `total_loss=`) — fixed in
`phase5_verify.py`.

### 3.2 Final Phase 5 result: PASS 9/9

5B / 4×H200 (of 8) / 24 fps × 121 f / 704×1280, rank 0, step-1 steady state:

| Metric | FULL | FULL_OFFLOAD | STREAMED_OFFLOAD |
|---|---:|---:|---:|
| P4_fwd_end max_alloc | 18.41 GB | 17.19 | 17.18 GB |
| P5_bwd_peak max_alloc | 23.40 GB | 23.33 | 23.38 GB |
| Step time (step 2) | 2.25 s | 2.32 s | 2.26 s (0.97× offload) |
| Loss step 2 | 4.37494 | 4.37379 | 4.37680 |
| bwd FILO 29→0 | ✓ | ✓ | ✓ |

Interpretation: at this scale the offload works (P4 drops by the entire
1.26 GB stash) but **cannot move the backward peak**, because (a) the stash
is ~3 % of the peak, and (b) the peak instant is at the *end* of backward
(gradients accumulate +0.167 GB/layer), where the stash is consumed in
every mode. Phase 5 therefore validates correctness, lifecycle, FILO
scheduling, and zero overhead — the GB win is Phase 6's claim.

### 3.3 Loss criterion restated (applies to both phases)

The plan's "loss identical within 1e-5" is unachievable on this stack:
backward is nondeterministic (attention backward + reduction order), and
FULL_OFFLOAD differs from *itself* at step 2 by ~8e-4 across same-seed
reruns (measured twice; FULL moved 1.2e-3 between harness runs). Mode-vs-
mode diffs (0.8–3.0e-3) sit inside that same-mode noise band, and hook-level
transparency is pinned separately by the unit suite (bit-exact pack/unpack
roundtrip; ULP-level wrapper backward). Phase 5/6 grade loss at
`LOSS_TOL=5e-3` ≈ 3–5× measured rerun noise.

## 4. Phase 6 — 14B / 8×H200 / rung 0 (2160×3840 × 77 f, sp=8)

648,000 DiT tokens (81 k/rank at sp=8), `num_latent_t=20`, synthetic
rung-0 parquet (8 samples, seed 42). 2-step probes per mode + 10-step
streamed soak. Verifier: PASS 9/9. Checkpoint saves disabled (one 14B
training-state checkpoint is ~86 GB).

### 4.1 Before / after (rank 0, step-2 steady state)

| Metric | FULL (before) | FULL_OFFLOAD | STREAMED_OFFLOAD (after) | Δ vs FULL |
|---|---:|---:|---:|---:|
| P4_fwd_end max_alloc | 80.06 GB | 47.71 | 47.71 GB | **−32.4 GB** |
| **P5_bwd_peak max_alloc** | **95.31 GB** | 70.29 | **71.13 GB** | **−24.2 GB (−25 %)** |
| P5 reserved | 107.4 GB | 76.2 | 77.9 GB | −29.5 GB |
| P7_opt_done max_alloc | 35.98 GB | 35.98 | 35.98 GB | — (Adam transient, out of scope) |
| Step time (step 2) | 627.9 s | 630.4 s | **628.2 s** | **+0.05 %** |
| Loss step 2 | 0.58423 | 0.58367 | 0.58553 | within noise (§3.3) |

The −32.4 GB at P4 is the entire 40-layer block-input stash
(40 × ~0.83 GB) moved off-GPU. Streamed vs naive offload at the peak is
+0.84 GB — exactly the one-deep in-flight prefetch buffer, as designed.

### 4.2 10-step streamed soak

- Step times 626.6–634.9 s, mean ≈ 628.9 s steady — **no upward ramp** (no
  CPU/pinned-memory fragmentation effect).
- P5 peak **flat at 71.13 GB on every steady step** (no allocator creep).
- All 10 losses finite; backward FILO 39→0 on all 40 blocks every step.

### 4.3 Acceptance vs plan §3 Phase 6

| Criterion | Result |
|---|---|
| Step time ≤ 1.05× FULL | ✅ 1.0005× — streaming is free |
| Loss match | ✅ within restated noise criterion |
| Peak ≤ 60 GB | ❌ 71.13 GB — missed by ~11 GB, gap attributed (§5) |

Note the measured FULL baseline (95.3 GB alloc / 107.4 reserved) is itself
well below the plan's ≈133 GB premise (different accounting/config in the
original estimate; the prior 14B report's 121 GB was whole-process
`nvidia-smi`). Both sides of the plan's ledger were overestimated.

## 5. Why 71 GB and not < 60 — gap attribution

§1 of the plan budgeted streamed peak ≈ 58.7 GB. Two assumptions fail:

1. **~11 GB of saved tensors live outside the wrapped blocks.** Offload-mode
   P4 allocated = 32.9 GB vs 21.7 GB static → ~11.2 GB of saved-for-backward
   tensors (patch/condition embedder outputs and other pre/post-block
   saves) that the per-block hook never sees. §1 assumed the *entire* stash
   streams; only the 40 block inputs (~33 GB at rung 0) do.
2. **Backward transient is ~38 GB, not the budgeted ~20 GB.** Peak occurs
   mid-backward at P4-base + transient (32.9 + 38.2 ≈ 71.1). The venv runs
   Torch SDPA, whose backward materialises much larger intermediates at
   81 k tokens/rank than the flash-attention-style budget assumed, on top
   of recompute working set and fp32 grad-shard accumulation.

### Follow-ups to close 71 → < 60 GB (priority order)

1. **Install flash-attn** (or force the flash SDPA kernel) — shrinks the
   dominant backward transient; likely the single biggest lever.
2. **Stream the non-block saves** (~11 GB): wrap embedder/pre-block stages
   or add a model-level saved-tensors hook.
3. **CPU-offload Adam state** (plan §7 stretch): ~−14 GB at measurable
   optimizer-step cost, if still short after 1–2.

Also note: at rung-0 step times (~628 s) the naive offload's synchronous
copies are already negligible (+0.4 %), so streaming's overlap advantage
will show on shorter-step / higher-bandwidth-pressure configs, not here.
Streaming's durable advantages over naive offload are the bounded prefetch
window and the copy-stream architecture the follow-ups build on.

## 6. Infrastructure findings

- **`hf download` stalls indefinitely on this box** (~1–2 MB/s with 0-byte
  periods; `.incomplete` files even shrank across resumes) while raw curl
  sustains ~70 MB/s/stream from the same CDN. Both models were fetched with
  `scripts/curl_hf_fetch.sh <org/repo>` (parallel curl + sha256 verification
  into the HF hub cache layout). See
  `.agents/lessons/2026-06-03_hf-download-stalls-use-curl.md`. Rule: measure
  the raw resource before building retry machinery around a slow client.
- The box's git deploy key is passphrase-protected → non-interactive fetch
  fails; an early Phase 5 run validated a stale checkout (pre-Phases-3&4)
  and had to be redone. Verify local == origin tip before long studies.
- 14B runs must disable training-state checkpointing
  (`training_state_checkpointing_steps 0`): one checkpoint ≈ 86 GB.

## 7. Artifact index

| Artifact | Path |
|---|---|
| Phase 5 final (PASS) | `probe_out/phase5_v2/{phase5_report.md,json, pytest.log, full*, streamed_offload}/` |
| Phase 5 first run (FAIL, pre-fix) | `probe_out/phase5/` |
| Phase 6 probes + verifier | `probe_out/phase6/{full,full_offload,streamed_offload}/`, `phase5_report.{md,json}` |
| Phase 6 soak | `probe_out/phase6/streamed_soak10/` |
| Datasets | `data/synthetic_5b_121f/` (5B rung 6), `data/synthetic_4k_14b_rung0/` |
| Scheduler fixes | `fastvideo/training/activation_streaming.py` (uncommitted on `peak-mem-reduction`) |
| Harness/verifier fixes | `scripts/4k_milestone/{phase5_validation.sh,phase5_verify.py,phase6_14b_study.sh}` |
| Plan doc updates | `memory_prefetch_plan.md` §10–§12 |
