# SPDX-License-Identifier: Apache-2.0
"""
Central registry for FastVideo pipelines and model configuration discovery.

This module mirrors the organization of sglang's registry while keeping
FastVideo's legacy behavior and mappings intact.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, Any

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
from fastvideo.api.sampling_param import SamplingParam

from fastvideo.fastvideo_args import WorkloadType
from fastvideo.logger import init_logger
from fastvideo.utils import (maybe_download_model_index, verify_model_config_and_directory)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
    from fastvideo.pipelines.pipeline_registry import PipelineType

# --- Part 1: Pipeline Discovery ---

_PIPELINE_REGISTRY: dict[str, dict[str, type[ComposedPipelineBase]]] = {}

# Registry for pipeline configuration classes (for single-file weights without
# model_index.json). Maps pipeline_class_name -> (PipelineConfig, SamplingParam)
_PIPELINE_CONFIG_REGISTRY: dict[str, tuple[type[PipelineConfig], type[SamplingParam]]] = {}


def _discover_and_register_pipelines() -> None:
    if _PIPELINE_REGISTRY:
        return

    from fastvideo.pipelines.pipeline_registry import import_pipeline_classes

    pipeline_classes = import_pipeline_classes()
    for pipeline_type, pipeline_dict in pipeline_classes.items():
        _PIPELINE_REGISTRY[pipeline_type] = pipeline_dict
        for pipeline_cls in pipeline_dict.values():
            if pipeline_cls is None:
                continue
            if hasattr(pipeline_cls, "pipeline_config_cls") and hasattr(pipeline_cls, "sampling_params_cls"):
                _PIPELINE_CONFIG_REGISTRY[pipeline_cls.__name__] = (
                    pipeline_cls.pipeline_config_cls,
                    pipeline_cls.sampling_params_cls,
                )


def get_pipeline_config_classes(pipeline_class_name: str) -> tuple[type[PipelineConfig], type[SamplingParam]] | None:
    _discover_and_register_pipelines()
    return _PIPELINE_CONFIG_REGISTRY.get(pipeline_class_name)


# --- Part 2: Config Registration ---


@dataclasses.dataclass
class ConfigInfo:
    """Encapsulates sampling + pipeline config classes for a model family."""

    sampling_param_cls: type[SamplingParam] | None
    pipeline_config_cls: type[PipelineConfig]
    workload_types: tuple[WorkloadType, ...]
    model_family: str | None = None
    default_preset: str | None = None


# The central registry mapping a model name to its configuration information
_CONFIG_REGISTRY: dict[str, ConfigInfo] = {}

# Mappings from Hugging Face model paths to our internal model names
_MODEL_HF_PATH_TO_NAME: dict[str, str] = {}

# Detectors to identify model families from paths or class names
_MODEL_NAME_DETECTORS: list[tuple[str, Callable[[str], bool]]] = []


def register_configs(
    sampling_param_cls: type[SamplingParam] | None,
    pipeline_config_cls: type[PipelineConfig],
    workload_types: tuple[WorkloadType, ...],
    hf_model_paths: list[str] | None = None,
    model_detectors: list[Callable[[str], bool]] | None = None,
    model_family: str | None = None,
    default_preset: str | None = None,
) -> None:
    """Register config classes for a model family.

    workload_types declares which UI workload options this config supports.
    Use () for configs not exposed as workload options.
    """
    model_id = str(len(_CONFIG_REGISTRY))

    _CONFIG_REGISTRY[model_id] = ConfigInfo(
        sampling_param_cls=sampling_param_cls,
        pipeline_config_cls=pipeline_config_cls,
        workload_types=workload_types,
        model_family=model_family,
        default_preset=default_preset,
    )

    if hf_model_paths:
        for path in hf_model_paths:
            if path in _MODEL_HF_PATH_TO_NAME:
                logger.warning("Model path '%s' is already mapped to '%s' and will be overwritten by '%s'.", path,
                               _MODEL_HF_PATH_TO_NAME[path], model_id)
            _MODEL_HF_PATH_TO_NAME[path] = model_id

    if model_detectors:
        for detector in model_detectors:
            _MODEL_NAME_DETECTORS.append((model_id, detector))


def get_model_short_name(model_id: str) -> str:
    if "/" in model_id:
        return model_id.split("/")[-1]
    return model_id


def _get_config_info(
    model_path: str,
    *,
    raise_on_missing: bool = True,
) -> ConfigInfo | None:
    # 1. Exact match
    if model_path in _MODEL_HF_PATH_TO_NAME:
        model_id = _MODEL_HF_PATH_TO_NAME[model_path]
        logger.debug("Resolved model path '%s' from exact path match.", model_path)
        return _CONFIG_REGISTRY.get(model_id)

    # 2. Partial match: use short model name.
    model_name = get_model_short_name(model_path.lower())
    all_model_hf_paths = sorted(_MODEL_HF_PATH_TO_NAME.keys(), key=len, reverse=True)
    for registered_model_hf_id in all_model_hf_paths:
        registered_model_name = get_model_short_name(registered_model_hf_id.lower())
        if registered_model_name == model_name:
            logger.debug("Resolved model name '%s' from partial path match.", registered_model_hf_id)
            model_id = _MODEL_HF_PATH_TO_NAME[registered_model_hf_id]
            return _CONFIG_REGISTRY.get(model_id)

    # 3. Use detectors (path or model_index pipeline name).
    if os.path.exists(model_path):
        config = verify_model_config_and_directory(model_path)
    else:
        config = maybe_download_model_index(model_path)

    pipeline_name = config.get("_class_name", "").lower()

    matched_model_names: list[str] = []
    for model_id, detector in _MODEL_NAME_DETECTORS:
        if detector(model_path.lower()) or detector(pipeline_name):
            logger.debug("Matched model name '%s' using a registered detector.", model_id)
            matched_model_names.append(model_id)

    if matched_model_names:
        if len(matched_model_names) > 1:
            logger.warning(
                "Multiple models matched for path '%s': %s. Using the first matched: '%s'.",
                model_path,
                matched_model_names,
                matched_model_names[0],
            )
        model_id = matched_model_names[0]
        return _CONFIG_REGISTRY.get(model_id)

    if raise_on_missing:
        raise RuntimeError(f"No model info found for model path: {model_path}")
    return None


def _register_configs() -> None:
    # Wan 2.1 — T2V (defaults provided by presets, no sampling_param_cls needed)
    register_configs(
        sampling_param_cls=None,
        pipeline_config_cls=WanT2V480PConfig,
        workload_types=(WorkloadType.T2V, ),
        hf_model_paths=[
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        ],
        model_detectors=[lambda path: "wanpipeline" in path.lower()],
        model_family="wan",
        default_preset="wan_t2v_1_3b",
    )
    register_configs(
        sampling_param_cls=None,
        pipeline_config_cls=WanT2V720PConfig,
        workload_types=(WorkloadType.T2V, ),
        hf_model_paths=[
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        ],
        model_family="wan",
        default_preset="wan_t2v_14b",
    )
    # Wan 2.1 — I2V
    register_configs(
        sampling_param_cls=None,
        pipeline_config_cls=WanI2V480PConfig,
        workload_types=(WorkloadType.I2V, ),
        hf_model_paths=[
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        ],
        model_detectors=[lambda path: "wanimagetovideo" in path.lower()],
        model_family="wan",
        default_preset="wan_i2v_14b_480p",
    )
    register_configs(
        sampling_param_cls=None,
        pipeline_config_cls=WanI2V720PConfig,
        workload_types=(WorkloadType.I2V, ),
        hf_model_paths=[
            "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        ],
        model_family="wan",
        default_preset="wan_i2v_14b_720p",
    )
    # Wan 2.2
    register_configs(
        sampling_param_cls=None,
        pipeline_config_cls=Wan2_2_TI2V_5B_Config,
        workload_types=(WorkloadType.T2V, WorkloadType.I2V),
        hf_model_paths=[
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        ],
        model_family="wan",
        default_preset="wan_2_2_ti2v_5b",
    )
    register_configs(
        sampling_param_cls=None,
        pipeline_config_cls=Wan2_2_T2V_A14B_Config,
        workload_types=(WorkloadType.T2V, ),
        hf_model_paths=[
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        ],
        model_family="wan",
        default_preset="wan_2_2_t2v_a14b",
    )
    register_configs(
        sampling_param_cls=None,
        pipeline_config_cls=Wan2_2_I2V_A14B_Config,
        workload_types=(WorkloadType.I2V, ),
        hf_model_paths=[
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        ],
        model_family="wan",
        default_preset="wan_2_2_i2v_a14b",
    )


# --- Part 3: Main Resolver ---


@dataclasses.dataclass
class ModelInfo:
    pipeline_cls: type[ComposedPipelineBase]
    sampling_param_cls: type[SamplingParam]
    pipeline_config_cls: type[PipelineConfig]


@lru_cache(maxsize=32)
def get_model_info(
    model_path: str,
    pipeline_type: PipelineType | str | None = None,
    workload_type: WorkloadType | None = None,
    override_pipeline_cls_name: str | None = None,
) -> ModelInfo:
    from fastvideo.pipelines.pipeline_registry import (PipelineType, get_pipeline_registry)

    if pipeline_type is None:
        pipeline_type = PipelineType.BASIC
    elif isinstance(pipeline_type, str):
        pipeline_type = PipelineType.from_string(pipeline_type)

    if workload_type is None:
        workload_type = WorkloadType.T2V

    if os.path.exists(model_path):
        config = verify_model_config_and_directory(model_path)
    else:
        config = maybe_download_model_index(model_path)

    pipeline_name = config.get("_class_name")
    if override_pipeline_cls_name:
        logger.info("Overriding pipeline class name from %s to %s", pipeline_name, override_pipeline_cls_name)
        pipeline_name = override_pipeline_cls_name

    if pipeline_name is None:
        raise ValueError("Model config does not contain a _class_name attribute. "
                         "Only diffusers format is supported.")

    pipeline_registry = get_pipeline_registry(pipeline_type)
    pipeline_cls = pipeline_registry.resolve_pipeline_cls(pipeline_name, pipeline_type, workload_type)

    config_info = _get_config_info(model_path, raise_on_missing=True)
    assert config_info is not None, "config_info must be resolved"

    sampling_param_cls = config_info.sampling_param_cls or SamplingParam

    return ModelInfo(
        pipeline_cls=pipeline_cls,
        sampling_param_cls=sampling_param_cls,
        pipeline_config_cls=config_info.pipeline_config_cls,
    )


def get_pipeline_config_cls_from_name(pipeline_name_or_path: str) -> type[PipelineConfig]:
    config_info = _get_config_info(pipeline_name_or_path, raise_on_missing=False)
    if config_info is None:
        raise ValueError(
            f"No match found for pipeline {pipeline_name_or_path}, please check the pipeline name or path.")
    return config_info.pipeline_config_cls


def get_sampling_param_cls_for_name(pipeline_name_or_path: str) -> Any | None:
    config_info = _get_config_info(pipeline_name_or_path, raise_on_missing=False)
    if config_info is None:
        logger.warning("No match found for pipeline %s, using default sampling param.", pipeline_name_or_path)
        return None
    return config_info.sampling_param_cls


_register_configs()


def _register_presets() -> None:
    from fastvideo.api.presets import register_preset
    from fastvideo.pipelines.basic.wan.presets import (
        ALL_PRESETS as WAN_PRESETS, )

    for preset in WAN_PRESETS:
        register_preset(preset)


_register_presets()


def get_model_family(model_path: str) -> str | None:
    """Return the ``model_family`` string for a model path, or ``None``."""
    config_info = _get_config_info(model_path, raise_on_missing=False)
    if config_info is None:
        return None
    return config_info.model_family


def get_default_preset(model_path: str) -> str | None:
    """Return the ``default_preset`` name for a model path."""
    config_info = _get_config_info(model_path, raise_on_missing=False)
    if config_info is None:
        return None
    return config_info.default_preset


def get_preset_selection(model_path: str) -> tuple[str | None, str | None]:
    """Return ``(default_preset, model_family)`` for a model path.

    Single-lookup variant of :func:`get_default_preset` +
    :func:`get_model_family`; callers that need both should prefer this
    to avoid walking the registry twice.
    """
    config_info = _get_config_info(model_path, raise_on_missing=False)
    if config_info is None:
        return None, None
    return config_info.default_preset, config_info.model_family


def get_registered_model_paths() -> list[str]:
    """Return all registered HuggingFace model paths.

    Useful for UIs and tooling that need to enumerate supported models.
    """
    return sorted(_MODEL_HF_PATH_TO_NAME.keys())


def get_registered_models_with_workloads(workload_type: str | None = None, ) -> list[dict[str, Any]]:
    """Return models with workload metadata, optionally filtered by workload.

    Args:
        workload_type: If set (e.g. "t2v", "i2v", "t2i"), only return models
            that support this workload. If None, return all with workload_types.

    Returns:
        List of dicts with keys: id, label, workload_types.
    """
    result: list[dict[str, Any]] = []
    for path in sorted(_MODEL_HF_PATH_TO_NAME.keys()):
        model_id = _MODEL_HF_PATH_TO_NAME[path]
        config_info = _CONFIG_REGISTRY.get(model_id)
        if config_info is None:
            continue
        workload_values = [w.value for w in config_info.workload_types]
        if workload_type is not None and workload_type.lower() not in workload_values:
            continue
        label = path.split("/")[-1].replace("-", " ").replace("_", " ")
        result.append({
            "id": path,
            "label": label,
            "workload_types": workload_values,
        })
    return result


__all__ = [
    "ConfigInfo",
    "ModelInfo",
    "get_default_preset",
    "get_model_family",
    "get_model_info",
    "get_pipeline_config_cls_from_name",
    "get_registered_model_paths",
    "get_registered_models_with_workloads",
    "get_sampling_param_cls_for_name",
    "get_pipeline_config_classes",
]
