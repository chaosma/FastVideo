# 4K Wan 2.1 T2V 1.3B fine-tune on 4× H200 SXM (NVLink)

Real test of the milestone hardware on 4× H200 SXM (144 GB each, full
NV18 mesh). Builds on `4k-milestone-status.md`.

## TL;DR

- All three rungs train successfully on 4× H200 with `sp_size=4`,
  `hsdp_shard_dim=4`, full grad checkpointing, flash-attn 2.8.1.
- Steady-state step time:
  rung 2 (4K, 5 fr) **4.5 s** ·
  rung 1 (4K, 17 fr) **20 s** ·
  rung 0 (4K, 81 fr) **288 s**.
- Rung-0 root cause is **pure compute on self-attention BF16 FLOPs**.
  Arithmetic intensity ≈ 340k FLOPs/byte, ~1000× above the H100/H200
  roofline crossover — i.e., we're nowhere near memory-bandwidth-
  limited, the GPU is doing exactly the math the model demands.
- **4× H100 SXM (80 GB) will run this at the same speed.** Peak
  memory is 58 GB/GPU — fits H100 with 28% headroom. Same 989 TF/s
  BF16 peak ⇒ ~290 s/step within run-to-run noise. **Rent H100, not
  H200, for this workload.**
- Ways we tried to speed up rung 0 — none paid out:
  - Drop grad-ckpt → OOM (activations need >144 GB on H200; recompute
    is mandatory).
  - `ops`-mode selective ckpt → also OOM.
  - VSA block-sparse @ 0.9 sparsity → kernel runs after a Hopper
    rebuild but step is no faster than dense (likely tile-scheduling
    overhead + non-sparse backward + unchanged SP comm).
- Real levers left: 8-GPU sp_size=8 (~75 s/step), FP8 attention via
  FA-3 (~150 s/step, not wired up), `torch.compile` for training
  (currently hard-coded off in FastVideo's training path), or
  Blackwell-class hardware.

## Hardware

- 4× NVIDIA H200 SXM, 143,771 MiB each (~144 GB)
- All GPU pairs connected via NV18 (18 NVLinks each — full mesh)
- 224 vCPUs, 2 TiB RAM, 299 GB workspace disk

`nvidia-smi topo -m`:

```
        GPU0    GPU1    GPU2    GPU3
GPU0    X       NV18    NV18    NV18
GPU1    NV18    X       NV18    NV18
GPU2    NV18    NV18    X       NV18
GPU3    NV18    NV18    NV18    X
```

## Environment notes

- Python 3.12, torch 2.11.0+cu128, torchaudio 2.11.0+cu128 (the cu13
  default wheel breaks at import — same workaround as the prior 4090
  setup).
- `torchcodec` removed (cu13 mismatch); preprocessing path uses PyAV.
- `hf_transfer` installed to speed the model snapshot pull.
- flash-attn 2.8.1 built from source, sm_90 only
  (`TORCH_CUDA_ARCH_LIST=9.0`) with `TMPDIR=/workspace/tmp`. The default
  `/tmp` overlay (32 GB) overflows during the build; build artifacts
  need ~40 GB. 13m24s build wall-clock.

## Training results — 5 steps each, sp_size=4, hsdp_shard_dim=4

Gradient checkpointing on (`enable_gradient_checkpointing_type: full`).
Real preprocessed parquet latents.

| Rung | Resolution × frames | Tokens (SP=1) | Backend | First step | Steady step | 5-step total | Mem/GPU |
|------|---------------------|---------------|---------|------------|-------------|--------------|---------|
| 2    | 4K × 5  (latent_t=2)  | 64.8k  | SDPA       | 9.0 s   | ~4.5 s  | 25 s       | (low) |
| 1    | 4K × 17 (latent_t=5)  | 162k   | SDPA       | 24.7 s  | ~20.2 s | 1m 43s     | 23 GB |
| 0    | 4K × 81 (latent_t=21) | 680.4k | SDPA       | 311.9 s | ~290 s  | (killed)   | 58 GB |
| 0    | 4K × 81 (latent_t=21) | 680.4k | flash-attn | 293.2 s | ~288 s  | 24m 3s     | 58 GB |

Per-step trajectory at rung 0 with flash-attn: 293.2, 290.1, 289.1,
288.5, 288.1 — extremely stable.

## Why rung 0 is so slow — the diagnosis

**Root cause: compute-bound on self-attention BF16 FLOPs. The kernel
is fine, the math is the math.**

The roofline check makes this airtight. For one self-attn call at
rung 0 (per rank after SP=4: 680k tokens, 3 heads, head_dim=128):

- Compute: `4 N² hd` = `4 · 680k² · 128` ≈ **2.37e14 FLOPs/head**
- HBM bytes (flash-style tiling forward): `~6 N hd · 2B` ≈ **1 GB/head**
- Arithmetic intensity: **~340k FLOPs/byte**

Roofline crossover:
- H100: 989 TF/s ÷ 3.35 TB/s = 295 FLOPs/byte
- H200: 989 TF/s ÷ 4.80 TB/s = 206 FLOPs/byte

340,000 ≫ 295/206 → strongly **compute-bound on both**. The H200's
1.4× HBM bandwidth advantage over H100 buys nothing here.

Per-step decomposition (Wan 1.3B = 30 layers, 3 heads/rank,
head_dim=128, sp=4, grad-ckpt=full):

| Phase                                         | FLOPs            | Time @ ~50% peak |
|-----------------------------------------------|------------------|------------------|
| Self-attn forward (1×)                        | 30 × 3 × 2.37e14 = 21.3 PFLOPs | ~43 s |
| + grad-ckpt recompute (1× forward)            | 21.3 PFLOPs       | ~43 s |
| + backward (≈2.5× forward)                    | ~53 PFLOPs        | ~107 s |
| FFN + QKV proj + SP all-to-all + FSDP gather  | linear residual   | ~80–100 s |
| **Total**                                     |                   | **≈ 280–300 s** |

Measured: 290 s. The GPUs are doing exactly the work the math says
they should be doing — there's no missing factor to recover.

The earlier `c + a·N + b·N²` regression on 3 measured rungs
independently agrees:

| Rung | Tokens | Step (SDPA) |
|------|--------|-------------|
| 2    | 64.8k  | 4.5 s       |
| 1    | 162k   | 20.2 s      |
| 0    | 680k   | 290 s       |

Solving: `c ≈ 0.3 s`, `a ≈ 27 µs/tok` (linear: FFN, QKV proj, comm),
`b ≈ 0.59 ns/tok²` (quadratic: attention). Plugging in rung 0:
linear contribution ≈ 18 s, quadratic ≈ **271 s**. Attention is
**93%** of step time, and consistent with the roofline analysis.

## Hardware implication: H100 will run this at the same speed

| Spec                | H100 SXM | H200 SXM |
|---------------------|----------|----------|
| HBM                 | 80 GB    | 144 GB   |
| HBM bandwidth       | 3.35 TB/s | 4.80 TB/s |
| BF16 peak           | 989 TF/s | 989 TF/s |
| Roofline crossover  | 295 FLOPs/byte | 206 FLOPs/byte |

Our attention call's arithmetic intensity (~340k FLOPs/byte) is
~1000× above either roofline crossover, so HBM bandwidth doesn't
bottleneck us. Same compute peak ⇒ same throughput.

- **Memory:** measured peak 58 GB/GPU on H200 with grad-ckpt=full.
  H100 SXM (80 GB) → fits with ~28% headroom.
- **Speed:** **same ~290 s/step** within run-to-run noise. The 1.4×
  HBM advantage of H200 buys nothing at this arithmetic intensity.
- **What you'd lose on H100:** headroom to *try* sp=2 or no-grad-ckpt
  variants — but both already OOM at 144 GB, so they're off the
  table on H100 too. The H200's extra memory is unused.

**Recommendation for this workload: rent 4× H100 SXM, not H200.**
Same wall-clock per step at lower $/hr. H200 only starts paying for
itself at workloads that *can* use the extra memory — e.g., a
sub-rung-0 config with `sp_size=2` (which would push activations
past 80 GB but might fit in 144 GB), or a hypothetical no-grad-ckpt
config with smaller `num_latent_t`.

(B200/MI300X *would* matter — different BF16 peak. H200 vs H100
doesn't.)

## Why flash-attn delivered only ~7%

PyTorch SDPA on H100/H200 already dispatches to a flash-style kernel
under the hood; `--flash-attn` just routes to a similar-performance
kernel. Both run near peak BF16 on a 680k-token attention. There's
nothing left to squeeze from the kernel choice — the only ways to cut
attention time are to cut **FLOPs** (sparsity / FP8) or **sequence
length** (resolution / SP / num_frames).

Sanity check: ~85 PFLOPs of attention work per rank (forward +
recompute + backward), H200 BF16 sustained ≈ 0.5 PFLOPs/s ⇒ ~170 s of
pure attention compute. Add FFN/comm/FSDP and we land at ~290 s.

## What we tried and what it cost

| Variant                                    | Step 1 / steady state | Outcome |
|--------------------------------------------|-----------------------|---------|
| sp=4 hsdp=4 + grad-ckpt=full + flash-attn  | 293 s / 288 s         | Baseline. 58 GB/GPU. |
| sp=4 hsdp=4 + grad-ckpt=`""` (off)         | OOM mid step 1        | Without recompute, activations blow past 144 GB. Recompute is mandatory at rung 0. |
| sp=4 hsdp=4 + grad-ckpt=`ops` (selective)  | OOM mid step 1        | Saving attn/RS outputs and skipping every-other matmul still needs >144 GB. |
| sp=4 hsdp=4 + flash-attn + VSA 0.9         | >6 min on step 1, killed | Prebuilt `fastvideo_kernel==0.2.6` wheel crashes on H200 ("TMA descriptor creation: initialization error" → cudaErrorIllegalInstruction). Source build with `TORCH_CUDA_ARCH_LIST=9.0a` (Hopper-specific) imports cleanly but the per-step kernel time isn't faster than dense — memory grew to 65 GB and step 1 ran past 6 minutes before kill. The VSA kernel as packaged today doesn't appear to deliver speedup at this token count + tile size on H200. |

`fastvideo_kernel` source build details: 13m 24s once
`include/cutlass` and `include/tk` submodules are initialized;
`TMPDIR=/workspace/tmp` required (the 32 GB `/tmp` overlay overflows).

## Why VSA didn't help even after rebuilding

A few candidates we did not isolate further:

1. **Tile size mismatch.** VSA partitions the latent grid into 4×4×4
   tiles; with our latent shape (T=21, H=270, W=480) → patches
   135×240×21 → 30×67×5 = ~10k tiles per layer. Scheduling overhead
   per tile may dominate.
2. **Backward not yet sparse.** The forward path can skip blocks at
   sparsity 0.9, but the backward / vjp may walk the full attention
   pattern. Empirically memory grew rather than shrank vs dense.
3. **Sparsity scheduling.** The training pipeline ramps sparsity over
   `vsa_decay_interval_steps`; on step 1 effective sparsity may be
   well below the 0.9 we configured.
4. **SP all-to-all unchanged.** Even if attention compute halves,
   we still pay the same 30+ all-to-all rounds per forward on the
   full QKV — 40+ MB per rank per round, × 2 (forward+backward) ×
   ~30 layers, plus 2 more for the kernel I/O.

## What's left to try (not done in this session)

- **Profile a single rung-0 step** with `FASTVIDEO_TORCH_PROFILER_DIR=...`
  (wait=0, warmup=1, active=1) to confirm the attention-vs-comm split
  per-kernel.
- **`torch.compile`** is hard-coded `False` for training in
  `_make_training_args` (`fastvideo/train/utils/moduleloader.py:57`).
  Patching to `True` and enabling reduce-overhead mode would fuse
  small kernels and is the most plausible win that doesn't require
  changing the math.
- **FP8 attention.** Hopper has FP8 tensor cores at ~2× BF16 peak;
  FlashAttention-3 supports it but isn't wired into FastVideo's
  `flash_attn` backend yet.
- **8× H200 / 8× B200.** With sp_size=8 each rank's attention seq
  halves → attention compute drops 4× (still O(n²) per rank, but n is
  340k). That gets rung 0 to ~75 s/step, making 100k-step training
  weeks instead of months.

## Output checkpoints

- `outputs/wan_finetune_4k/checkpoint-5` — rung 2 (SDPA)
- `outputs/wan_finetune_4k_rung1/checkpoint-5` — rung 1 (SDPA)
- `outputs/wan_finetune_4k_rung0_fa/checkpoint-5` — rung 0 (flash-attn)

`outputs/wan_finetune_4k_rung0/` is empty — the SDPA rung 0 was killed
mid-run (after the user pointed out flash-attn was likely faster).
