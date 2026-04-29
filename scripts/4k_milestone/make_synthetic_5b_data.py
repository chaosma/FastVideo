"""
Synthetic parquet generator for the Wan 2.2 TI2V 5B model.

Why a separate script: the 5B model uses Wan2.2's VAE which differs from
the 1.3B's. Latent shape changes:

  Wan2.1-T2V-1.3B:    z_dim=16,  spatial_stride= 8  → [16, T, H/8,  W/8]
  Wan2.2-TI2V-5B:     z_dim=48,  spatial_stride=16  → [48, T, H/16, W/16]

The 5B DiT has patch_size=(1, 2, 2), so latent spatial dims must be even.
2160 / 16 = 135 (ODD) — fails inside the patch embed. Round source video
height up to 2176 (= 136 × 16) before preprocessing.

Usage:
    python scripts/4k_milestone/make_synthetic_5b_data.py [--rung N] [--n-samples N]

After running, train with:
    bash examples/train/run.sh \\
        examples/train/configs/fine_tuning/wan/t2v_4k.yaml \\
        --models.student.init_from Wan-AI/Wan2.2-TI2V-5B-Diffusers \\
        --pipeline.flow_shift 5.0 \\
        --training.distributed.num_gpus 4 \\
        --training.distributed.sp_size 4 \\
        --training.distributed.hsdp_shard_dim 4 \\
        --training.data.data_path data/synthetic_4k_5b \\
        --training.data.num_height 2176 \\
        --training.data.num_width 3840 \\
        --training.data.num_frames 5 \\
        --training.data.num_latent_t 2 \\
        --training.loop.max_train_steps 10
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ── Rung definitions for 5B (heights chosen so latent dims are even) ──────────
# Latent shape: [C=48, T=num_latent_t, H=height//16, W=width//16]
RUNGS = {
    0: dict(height=2176, width=3840, num_latent_t=20, num_frames=77),
    1: dict(height=2176, width=3840, num_latent_t=5,  num_frames=17),
    2: dict(height=2176, width=3840, num_latent_t=2,  num_frames=5),
    3: dict(height=2176, width=3840, num_latent_t=1,  num_frames=1),
    4: dict(height=1440, width=2560, num_latent_t=1,  num_frames=1),
    5: dict(height=1088, width=1920, num_latent_t=1,  num_frames=1),
}

# Wan 2.2 VAE config
VAE_SPATIAL_STRIDE = 16
VAE_Z_DIM = 48

# Text embedding (UMT5)
TEXT_DIM = 4096
TEXT_SEQ_LEN = 13

CAPTIONS = [
    "A 4K ultra-high-definition waterfall (5B synth, copy 1).",
    "A 4K ultra-high-definition forest canopy (5B synth, copy 2).",
    "A 4K ultra-high-definition ocean waves (5B synth, copy 3).",
    "A 4K ultra-high-definition mountain peaks (5B synth, copy 4).",
]


def make_schema() -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("vae_latent_bytes", pa.large_binary()),
        pa.field("vae_latent_shape", pa.list_(pa.int64())),
        pa.field("vae_latent_dtype", pa.string()),
        pa.field("text_embedding_bytes", pa.large_binary()),
        pa.field("text_embedding_shape", pa.list_(pa.int64())),
        pa.field("text_embedding_dtype", pa.string()),
        pa.field("file_name", pa.string()),
        pa.field("caption", pa.string()),
        pa.field("media_type", pa.string()),
        pa.field("width", pa.int64()),
        pa.field("height", pa.int64()),
        pa.field("num_frames", pa.int64()),
        pa.field("duration_sec", pa.float64()),
        pa.field("fps", pa.float64()),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rung", type=int, default=2, choices=list(RUNGS.keys()))
    parser.add_argument("--n-samples", type=int, default=4,
                        help="Must be >= num_gpus for dataset.shard to work")
    parser.add_argument("--output-dir", type=str, default="data/synthetic_4k_5b")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = RUNGS[args.rung]
    h_lat = cfg["height"] // VAE_SPATIAL_STRIDE
    w_lat = cfg["width"] // VAE_SPATIAL_STRIDE
    t_lat = cfg["num_latent_t"]
    if h_lat % 2 or w_lat % 2:
        raise ValueError(
            f"Latent dims must be even for patch_size=2; got {h_lat}x{w_lat}")

    latent_shape = [VAE_Z_DIM, t_lat, h_lat, w_lat]
    text_shape = [TEXT_SEQ_LEN, TEXT_DIM]
    print(f"Rung {args.rung}: {cfg['width']}×{cfg['height']}, {cfg['num_frames']} frames")
    print(f"Latent shape: {latent_shape} (5B Wan2.2 VAE)")
    print(f"Text shape:   {text_shape}")

    rng = np.random.default_rng(args.seed)
    rows = []
    for i in range(args.n_samples):
        latent = rng.standard_normal(latent_shape).astype(np.float32)
        text = rng.standard_normal(text_shape).astype(np.float32)
        rows.append({
            "id": f"synth_5b_{i:04d}",
            "vae_latent_bytes": latent.tobytes(),
            "vae_latent_shape": latent_shape,
            "vae_latent_dtype": "float32",
            "text_embedding_bytes": text.tobytes(),
            "text_embedding_shape": text_shape,
            "text_embedding_dtype": "float32",
            "file_name": f"synth_5b_{i:04d}.mp4",
            "caption": CAPTIONS[i % len(CAPTIONS)],
            "media_type": "video",
            "width": cfg["width"],
            "height": cfg["height"],
            "num_frames": cfg["num_frames"],
            "duration_sec": float(cfg["num_frames"]) / 16.0,
            "fps": 16.0,
        })

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"synthetic_4k_5b_rung{args.rung}.parquet")
    pq.write_table(pa.Table.from_pylist(rows, schema=make_schema()), out)
    print(f"Wrote {out} ({os.path.getsize(out)/1e6:.1f} MB, {args.n_samples} rows)")


if __name__ == "__main__":
    main()
