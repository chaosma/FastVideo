"""
Verify a parquet dataset produced by preprocess_4k.sh or make_synthetic_4k_data.py.

Usage:
    python scripts/4k_milestone/verify_parquet.py <parquet_dir_or_file>

Prints:
  - Number of rows
  - Latent shape and dtype for each row
  - Text embedding shape
  - Caption text
  - File size
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pyarrow.parquet as pq


def verify(path: str) -> None:
    if os.path.isdir(path):
        files = sorted(
            os.path.join(path, f) for f in os.listdir(path)
            if f.endswith(".parquet")
        )
        if not files:
            # try subdirectory (combined_parquet_dataset)
            for sub in os.listdir(path):
                subdir = os.path.join(path, sub)
                if os.path.isdir(subdir):
                    files += sorted(
                        os.path.join(subdir, f) for f in os.listdir(subdir)
                        if f.endswith(".parquet")
                    )
        if not files:
            print(f"No .parquet files found under {path}")
            sys.exit(1)
    else:
        files = [path]

    total_rows = 0
    for fpath in files:
        size_mb = os.path.getsize(fpath) / 1e6
        table = pq.read_table(fpath)
        n = len(table)
        total_rows += n
        print(f"\n{'='*60}")
        print(f"File: {fpath}  ({size_mb:.1f} MB, {n} rows)")
        print(f"Schema fields: {table.schema.names}")

        for i in range(min(n, 3)):  # show first 3 rows
            row = {col: table.column(col)[i].as_py() for col in table.schema.names}
            print(f"\n  Row {i}:")

            if "vae_latent_bytes" in row and "vae_latent_shape" in row:
                shape = row["vae_latent_shape"]
                raw = row["vae_latent_bytes"]
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
                n_elem = arr.size
                size_mb_latent = arr.nbytes / 1e6
                print(f"    vae_latent: shape={shape}  ({size_mb_latent:.1f} MB fp32)")
                print(f"    vae_latent: mean={arr.mean():.4f}  std={arr.std():.4f}  "
                      f"min={arr.min():.4f}  max={arr.max():.4f}")
                # Interpret shape
                if len(shape) == 4:
                    C, T, H, W = shape
                    spatial_tokens = (H // 2) * (W // 2)  # patch_size=2
                    total_tokens = spatial_tokens * T
                    print(f"    vae_latent: C={C} T={T} H={H} W={W} → "
                          f"~{total_tokens:,} DiT tokens (patch_size=2×2)")

            if "text_embedding_bytes" in row and "text_embedding_shape" in row:
                shape = row["text_embedding_shape"]
                size_mb_emb = (np.prod(shape) * 4) / 1e6
                print(f"    text_embedding: shape={shape}  ({size_mb_emb:.2f} MB fp32)")

            if "caption" in row:
                cap = str(row["caption"])[:120]
                print(f"    caption: {cap!r}")

    print(f"\n{'='*60}")
    print(f"Total rows across {len(files)} file(s): {total_rows}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_parquet.py <parquet_dir_or_file>")
        sys.exit(1)
    verify(sys.argv[1])
