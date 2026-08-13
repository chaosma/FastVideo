# SPDX-License-Identifier: Apache-2.0
"""Runtime metadata for tensors produced inside transformer blocks."""
from __future__ import annotations

import contextlib
import contextvars
import functools
from collections.abc import Iterator
from typing import Any

import torch

_CURRENT_BLOCK_INDEX: contextvars.ContextVar[int | None] = (
    contextvars.ContextVar("fastvideo_current_block_index", default=None))


def current_block_index() -> int | None:
    """Current global transformer-block index, when executing inside a block."""
    return _CURRENT_BLOCK_INDEX.get()


@contextlib.contextmanager
def block_context(block_index: int | None) -> Iterator[None]:
    token = _CURRENT_BLOCK_INDEX.set(block_index)
    try:
        yield
    finally:
        _CURRENT_BLOCK_INDEX.reset(token)


def wrap_forward_with_block_context(module: torch.nn.Module,
                                    block_index: int) -> torch.nn.Module:
    """Tag ``module.forward`` execution with a global block index."""
    if getattr(module, "_fastvideo_block_context_index", None) == block_index:
        return module

    forward = module.forward

    @functools.wraps(forward)
    def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
        with block_context(block_index):
            return forward(*args, **kwargs)

    module.forward = wrapped_forward  # type: ignore[method-assign]
    module._fastvideo_block_context_index = block_index  # type: ignore[attr-defined]
    return module
