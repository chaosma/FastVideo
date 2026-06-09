# SPDX-License-Identifier: Apache-2.0
"""Stub kept temporarily for non-Wan pipelines that will be removed.

Originally this module bridged the legacy ``GenerationRequest``/serve
schema with the kwargs accepted by the inference engines. After the
inference layer was deleted, only ``register_continuation_kind`` is
still referenced (by ``pipelines/basic/ltx2/continuation.py``); that
caller is removed in the non-Wan pipeline cleanup.
"""
from __future__ import annotations

from typing import Any


def register_continuation_kind(*_args: Any, **_kwargs: Any) -> None:
    """No-op stub. Continuation registration is not used by Wan."""
