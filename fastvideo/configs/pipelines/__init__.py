# SPDX-License-Identifier: Apache-2.0
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.configs.pipelines.wan import (
    Wan2_2_I2V_A14B_Config,
    Wan2_2_T2V_A14B_Config,
    Wan2_2_TI2V_5B_Config,
    WanI2V480PConfig,
    WanI2V720PConfig,
    WanT2V480PConfig,
    WanT2V720PConfig,
)
from fastvideo.registry import get_pipeline_config_cls_from_name

__all__ = [
    "PipelineConfig",
    "Wan2_2_I2V_A14B_Config",
    "Wan2_2_T2V_A14B_Config",
    "Wan2_2_TI2V_5B_Config",
    "WanI2V480PConfig",
    "WanI2V720PConfig",
    "WanT2V480PConfig",
    "WanT2V720PConfig",
    "get_pipeline_config_cls_from_name",
]
