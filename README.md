# FastVideo (minimal Wan fork)

A trimmed-down fork of [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo)
that keeps only the **parquet preprocessing** and **YAML-driven training**
paths for the **Wan** model family (Wan 2.1 T2V/I2V 1.3B + 14B, Wan 2.2
TI2V 5B, Wan 2.2 T2V/I2V A14B). Inference servers, ComfyUI, web UI,
LoRA, distillation, validation, VSA, and every other model family
(Hunyuan, LTX-2, Cosmos, Gen3C, HYWorld, LongCat, MatrixGame, SD3.5,
TurboDiffusion, …) are removed.

If you need any of those, use upstream FastVideo.

## Install

```bash
uv pip install -e .
# or:
pip install -e .
```

Optional dev/lint extras:

```bash
uv pip install -e .[dev]
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

## Preprocess 4K video → parquet latents

```bash
# Place raw clips in data/raw_4k/ along with a metadata.csv
# (file_path,caption per row).
bash scripts/4k_milestone/preprocess_4k.sh --rung 2
# Output: data/preprocessed_4k/training_dataset/worker_*/...
```

Rung map (number of source frames; latent timesteps satisfy `(N-1) % 4 == 0`):

| Rung | Frames | Latent t | Notes |
|------|--------|----------|-------|
| 0    | 77     | 20       | Full Wan 2.1 native length |
| 1    | 17     | 5        | |
| 2    | 5      | 2        | Default; cheapest smoke test |
| 3    | 1      | 1        | Image mode |

For Wan 2.2 5B (16× spatial VAE) override the model:

```bash
MODEL_PATH=Wan-AI/Wan2.2-TI2V-5B-Diffusers \
MAX_HEIGHT=2176 \
bash scripts/4k_milestone/preprocess_4k.sh --rung 2
```

## Train

```bash
NUM_GPUS=4 bash examples/train/run.sh \
    examples/train/configs/fine_tuning/wan/t2v_4k.yaml \
    --training.data.data_path data/preprocessed_4k/training_dataset \
    --training.distributed.sp_size 4 \
    --training.distributed.hsdp_shard_dim 4
```

Quick dry-run (no GPU step, just config parse + dataloader build):

```bash
bash examples/train/run.sh \
    examples/train/configs/fine_tuning/wan/t2v_4k.yaml \
    --dry-run --training.distributed.num_gpus 1
```

The trainer entry is `fastvideo/train/entrypoint/train.py` and the only
training method kept is `FineTuneMethod`
(`fastvideo.train.methods.fine_tuning.finetune`). Checkpoints land in
`outputs/<run_name>/checkpoint-<step>/dcp/` (DCP-sharded).

To convert a DCP shard set into a Diffusers checkpoint:

```bash
python -m fastvideo.train.entrypoint.dcp_to_diffusers \
    --config examples/train/configs/fine_tuning/wan/t2v_4k.yaml \
    --checkpoint-dir outputs/<run_name>/checkpoint-<step>
```

## Tests

```bash
pytest tests/             # imports + dry-run smoke + CLI surface
pytest -k 'dry_run'       # the dry-run test alone (uses data/real_4k_rung2)
```

## Inference

A trimmed CLI is included for sampling from a checkpoint:

```bash
fastvideo generate --config <run.yaml> [--request.prompt "..."]
# or directly:
python -m fastvideo.entrypoints.cli.main generate --config <run.yaml>
```

The Python API is also available:

```python
from fastvideo import VideoGenerator
gen = VideoGenerator.from_config(run_config.generator)
gen.generate(run_config.request)
```

The OpenAI-style HTTP server, streaming server, and Ray executor are NOT
included. Inference uses the multiprocess executor (`mp` backend) only.
LoRA at inference is also not wired up.

## Repository layout

```
fastvideo/
├── api/              # config schemas + CLI parser (shared by inference + training)
├── attention/        # SDPA / FlashAttention selectors
├── configs/          # Wan-only model + pipeline configs
├── dataset/          # parquet map-style dataloader + preprocessing datasets
├── distributed/      # SP / FSDP / NCCL helpers
├── entrypoints/      # PURE INFERENCE: CLI + VideoGenerator
├── hooks/, layers/, logging_utils/, platforms/, third_party/
├── models/
│   ├── dits/wanvideo.py      # WanTransformer3DModel
│   ├── vaes/{wanvae,common,autoencoder_kl}.py
│   ├── encoders/{t5,t5_hf,clip,vision,base}.py
│   ├── schedulers/{flow_match_euler_discrete,flow_unipc_multistep,unipc_multistep}.py
│   └── loader/, registry.py
├── pipelines/
│   ├── basic/wan/{wan_pipeline,wan_i2v_pipeline,presets}.py  # T2V + I2V (used by both)
│   ├── preprocess/wan/wan_preprocess_pipelines.py + v1_preprocessing_new.py
│   └── stages/                                               # 11 Wan-relevant stages
├── train/                   # PURE TRAINING (YAML-driven)
│   ├── entrypoint/train.py + dcp_to_diffusers.py
│   ├── methods/fine_tuning/finetune.py
│   ├── models/wan/wan.py
│   └── callbacks/{ema,grad_clip}.py + utils/
├── training/                # SHARED trainer utilities (training_utils,
│                            # activation_checkpoint, checkpointing_utils,
│                            # trackers) — the *_pipeline.py classes were dropped
├── worker/                  # PURE INFERENCE: multiproc GPU worker pool
└── workflow/preprocess/     # preprocess workflow orchestration
examples/train/configs/fine_tuning/wan/{t2v,t2v_4k}.yaml
scripts/4k_milestone/        # preprocess_4k.sh + synthetic data generators
scripts/checkpoint_conversion/wan_to_diffusers.py
reports/                     # 4K H200 + 4K Wan2.2-5B benchmark writeups
```

The cleanest train/inference separation:

- **Training-only:** `fastvideo/train/`
- **Inference-only:** `fastvideo/entrypoints/`, `fastvideo/worker/`
- **Shared:** `fastvideo/api/` (full config schema), `fastvideo/pipelines/basic/wan/` (training also calls WanPipeline on rank 0 to encode the negative prompt at startup), `fastvideo/training/` (utilities), the model graph in `fastvideo/models/`

## What's intentionally missing

- OpenAI-style HTTP server, streaming server, gradio demo, ComfyUI
  nodes, web UI. Inference is via the `fastvideo generate` CLI or the
  `VideoGenerator` Python API only.
- Ray executor — inference uses the multiprocess executor only.
- LoRA at inference / VSA / distillation (DMD, KD, self-forcing, dfsft).
- Validation during training: `ValidationCallback` was removed because
  the canonical 4K configs ran with it disabled.
- All non-Wan model families. `fastvideo/models/dits/` keeps only
  `wanvideo.py` (+ `base.py`).

## Hardware notes

The `reports/` directory has detailed 4K results:

- **Wan 2.1 T2V 1.3B on 4× H200**: rung 2 ≈ 4.5 s/step, rung 1 ≈ 20 s,
  rung 0 ≈ 288 s. Compute-bound on attention; H100 SXM matches H200 at
  this arithmetic intensity. Peak memory 58 GB/GPU at rung 0.
- **Wan 2.2 TI2V 5B on 4× H200**: rung 2 ≈ 2.15 s/step, rung 1 ≈ 4.89 s,
  rung 0 ≈ 40.6 s. The 16× spatial VAE buys 4× fewer DiT tokens, so the
  larger 5B is ~7× faster than 1.3B at rung 0.

For rung-2 smoke tests on a 4× RTX 4090 box, the 1.3B model fits and
runs in single-digit seconds per step.
