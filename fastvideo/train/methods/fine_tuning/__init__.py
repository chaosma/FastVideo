# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod

__all__ = [
    "FineTuneMethod",
]


def __getattr__(name: str) -> object:
    if name == "FineTuneMethod":
        from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod
        return FineTuneMethod
    raise AttributeError(name)
