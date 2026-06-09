# SPDX-License-Identifier: Apache-2.0
"""Public API surface kept by the minimal Wan-only refactor.

Only the symbols that the parquet-preprocessing and YAML-driven
training paths still import are re-exported here. The full inference
schema/parser/server surface has been removed.
"""
from fastvideo.api.errors import ConfigValidationError
from fastvideo.api.presets import (
    InferencePreset,
    PresetStageSpec,
    get_all_preset_names,
    get_preset,
    get_presets_for_family,
    register_preset,
    validate_preset_selection,
    validate_stage_names,
    validate_stage_overrides,
)
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.api.schema import ContinuationState

__all__ = [
    "ConfigValidationError",
    "ContinuationState",
    "InferencePreset",
    "PresetStageSpec",
    "SamplingParam",
    "get_all_preset_names",
    "get_preset",
    "get_presets_for_family",
    "register_preset",
    "validate_preset_selection",
    "validate_stage_names",
    "validate_stage_overrides",
]
