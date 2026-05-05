# SPDX-License-Identifier: Apache-2.0
"""Smoke-test the kept entrypoints can be imported without GPU."""

from __future__ import annotations


def test_train_entrypoint_imports() -> None:
    from fastvideo.train.entrypoint import train  # noqa: F401


def test_preprocess_entrypoint_imports() -> None:
    from fastvideo.pipelines.preprocess import v1_preprocessing_new  # noqa: F401


def test_wan_pipelines_import() -> None:
    from fastvideo.pipelines.basic.wan.wan_pipeline import WanPipeline  # noqa: F401
    from fastvideo.pipelines.basic.wan.wan_i2v_pipeline import (  # noqa: F401
        WanImageToVideoPipeline, )


def test_finetune_method_imports() -> None:
    from fastvideo.train.methods.fine_tuning.finetune import (  # noqa: F401
        FineTuneMethod, )
    from fastvideo.train.models.wan import WanModel  # noqa: F401


def test_wan_pipeline_configs_import() -> None:
    from fastvideo.configs.pipelines import (  # noqa: F401
        Wan2_2_I2V_A14B_Config,
        Wan2_2_T2V_A14B_Config,
        Wan2_2_TI2V_5B_Config,
        WanI2V480PConfig,
        WanI2V720PConfig,
        WanT2V480PConfig,
        WanT2V720PConfig,
    )
