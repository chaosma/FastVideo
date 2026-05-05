# SPDX-License-Identifier: Apache-2.0
"""End-to-end --dry-run check on the canonical 4K Wan T2V config.

The trainer's --dry-run mode parses the YAML, instantiates the role
models, builds the dataloader, and exits before the first forward pass.
This catches dependency / config drift without requiring a GPU step.

Skipped if no parquet rung-2 dataset is present in the workspace.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", "real_4k_rung2")
CONFIG = os.path.join(
    REPO_ROOT,
    "examples",
    "train",
    "configs",
    "fine_tuning",
    "wan",
    "t2v_4k.yaml",
)


@pytest.mark.skipif(
    not os.path.isdir(DATA_PATH),
    reason=f"rung-2 4K parquet not found at {DATA_PATH}",
)
def test_train_dry_run_rung2_4k() -> None:
    """Trainer dry-run against rung-2 4K parquet completes successfully."""

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "--master_port",
        "29555",
        "fastvideo/train/entrypoint/train.py",
        "--config",
        CONFIG,
        "--dry-run",
        "--training.distributed.num_gpus",
        "1",
        "--training.data.data_path",
        DATA_PATH,
    ]
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    combined = completed.stdout + completed.stderr
    assert "Dry-run: config parsed and build_from_config succeeded." in combined, (
        f"dry-run did not report success.\nstdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}")
