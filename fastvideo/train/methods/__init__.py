# SPDX-License-Identifier: Apache-2.0

from fastvideo.train.methods.base import TrainingMethod

__all__ = [
    "TrainingMethod",
    "FineTuneMethod",
]


def __getattr__(name: str) -> object:
    if name == "FineTuneMethod":
        from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod
        return FineTuneMethod
    raise AttributeError(name)
