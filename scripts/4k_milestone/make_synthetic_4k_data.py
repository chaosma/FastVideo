"""
Rung 6 — CPU smoke test: create a synthetic parquet dataset with 4K-shaped latents.

This lets you verify the full training plumbing (config loading, dataloader,
model forward pass shape math) on a machine without CUDA, without running
real preprocessing.

Usage:
    python scripts/4k_milestone/make_synthetic_4k_data.py [--rung N] [--n-samples N]

Output:
    data/synthetic_4k/  — parquet files readable by the FastVideo dataloader

After running, dry-run the training config:
    bash examples/train/run.sh \\
        examples/train/configs/fine_tuning/wan/t2v_4k.yaml \\
        --dry-run \\
        --training.distributed.num_gpus 1 \\
        --training.data.data_path data/synthetic_4k

NOTE: This is a plumbing check only — the latent values are random noise.
      Real training requires real preprocessed latents from preprocess_4k.sh.
"""

from __future__ import annotations

import argparse
import os
import struct
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ── Rung definitions ──────────────────────────────────────────────────────────
# Latent shape: [C=16, T=num_latent_t, H=height//8, W=width//8]
RUNGS = {
    0: dict(height=2160, width=3840, num_latent_t=21, num_frames=81),
    1: dict(height=2160, width=3840, num_latent_t=5,  num_frames=17),
    2: dict(height=2160, width=3840, num_latent_t=2,  num_frames=5),
    3: dict(height=2160, width=3840, num_latent_t=1,  num_frames=1),
    4: dict(height=1440, width=2560, num_latent_t=1,  num_frames=1),
    5: dict(height=1080, width=1920, num_latent_t=1,  num_frames=1),
}

# Wan VAE config
VAE_SPATIAL_STRIDE = 8
VAE_Z_DIM = 16

# Text embedding: UMT5 → hidden_dim=4096; pad to 512 tokens (matches dataloader default)
TEXT_DIM = 4096
TEXT_SEQ_LEN = 77  # actual sequence; loader pads to 512

CAPTIONS = [
    "A 4K ultra-high-definition waterfall cascading over mossy rocks.",
    "Aerial view of a dense forest canopy at golden hour in 4K.",
    "Close-up of ocean waves crashing on a rocky shore, ultra-high resolution.",
    "Time-lapse of clouds moving over mountain peaks in 4K.",
]


def tensor_to_bytes(arr: np.ndarray) -> bytes:
    """Serialize a float32 numpy array to raw bytes (matches dataloader frombuffer)."""
    return arr.astype(np.float32).tobytes()


def make_schema() -> pa.Schema:
    return pa.schema([
        pa.field("vae_latent_bytes", pa.large_binary()),
        pa.field("vae_latent_shape", pa.list_(pa.int64())),
        pa.field("vae_latent_dtype", pa.string()),
        pa.field("text_embedding_bytes", pa.large_binary()),
        pa.field("text_embedding_shape", pa.list_(pa.int64())),
        pa.field("text_embedding_dtype", pa.string()),
        pa.field("caption", pa.string()),
    ])


def make_sample(rng: np.random.Generator, cfg: dict, caption: str) -> dict:
    h_latent = cfg["height"] // VAE_SPATIAL_STRIDE
    w_latent = cfg["width"] // VAE_SPATIAL_STRIDE
    t_latent = cfg["num_latent_t"]

    # Latent: [C, T, H, W]
    latent_shape = [VAE_Z_DIM, t_latent, h_latent, w_latent]
    latent = rng.standard_normal(latent_shape).astype(np.float32)

    # Text embedding: [seq_len, text_dim]
    text_shape = [TEXT_SEQ_LEN, TEXT_DIM]
    text_emb = rng.standard_normal(text_shape).astype(np.float32)

    return {
        "vae_latent_bytes": tensor_to_bytes(latent),
        "vae_latent_shape": latent_shape,
        "vae_latent_dtype": "float32",
        "text_embedding_bytes": tensor_to_bytes(text_emb),
        "text_embedding_shape": text_shape,
        "text_embedding_dtype": "float32",
        "caption": caption,
    }


def write_parquet(samples: list[dict], path: str, schema: pa.Schema) -> None:
    table = pa.table(
        {
            "vae_latent_bytes":     pa.array([s["vae_latent_bytes"] for s in samples], type=pa.large_binary()),
            "vae_latent_shape":     pa.array([s["vae_latent_shape"] for s in samples], type=pa.list_(pa.int64())),
            "vae_latent_dtype":     pa.array([s["vae_latent_dtype"] for s in samples], type=pa.string()),
            "text_embedding_bytes": pa.array([s["text_embedding_bytes"] for s in samples], type=pa.large_binary()),
            "text_embedding_shape": pa.array([s["text_embedding_shape"] for s in samples], type=pa.list_(pa.int64())),
            "text_embedding_dtype": pa.array([s["text_embedding_dtype"] for s in samples], type=pa.string()),
            "caption":              pa.array([s["caption"] for s in samples], type=pa.string()),
        },
        schema=schema,
    )
    pq.write_table(table, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rung", type=int, default=2, choices=list(RUNGS.keys()),
                        help="Rung to generate data for (default: 2 = 4K 5 frames)")
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Number of synthetic samples to generate (default: 8)")
    parser.add_argument("--output-dir", type=str, default="data/synthetic_4k",
                        help="Directory to write parquet files (default: data/synthetic_4k)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = RUNGS[args.rung]
    h_latent = cfg["height"] // VAE_SPATIAL_STRIDE
    w_latent = cfg["width"] // VAE_SPATIAL_STRIDE
    latent_mb = (VAE_Z_DIM * cfg["num_latent_t"] * h_latent * w_latent * 4) / 1e6

    print(f"Generating {args.n_samples} synthetic samples")
    print(f"  Rung {args.rung}: {cfg['width']}×{cfg['height']}, {cfg['num_frames']} frames")
    print(f"  Latent shape:  [{VAE_Z_DIM}, {cfg['num_latent_t']}, {h_latent}, {w_latent}]")
    print(f"  Latent size:   {latent_mb:.1f} MB per sample (float32)")
    print(f"  Output:        {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    schema = make_schema()
    rng = np.random.default_rng(args.seed)

    samples = []
    for i in range(args.n_samples):
        caption = CAPTIONS[i % len(CAPTIONS)]
        sample = make_sample(rng, cfg, caption)
        samples.append(sample)
        print(f"  Generated sample {i+1}/{args.n_samples}", end="\r")

    print()

    # Write to a single parquet file
    out_path = os.path.join(args.output_dir, "synthetic_4k_rung{}.parquet".format(args.rung))
    write_parquet(samples, out_path, schema)
    print(f"Written: {out_path}")
    print()
    print("To verify the parquet, run:")
    print(f"  python scripts/4k_milestone/verify_parquet.py {args.output_dir}")
    print()
    print("To dry-run the training config:")
    print(f"  bash examples/train/run.sh \\")
    print(f"      examples/train/configs/fine_tuning/wan/t2v_4k.yaml \\")
    print(f"      --dry-run \\")
    print(f"      --training.distributed.num_gpus 1 \\")
    print(f"      --training.data.data_path {args.output_dir}")


if __name__ == "__main__":
    main()
