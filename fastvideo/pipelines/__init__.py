# SPDX-License-Identifier: Apache-2.0
"""Diffusion pipelines for fastvideo."""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch, TrainingBatch
from fastvideo.pipelines.pipeline_registry import PipelineType
from fastvideo.registry import get_model_info
from fastvideo.utils import maybe_download_model

logger = init_logger(__name__)


def build_pipeline(
    fastvideo_args: FastVideoArgs,
    pipeline_type: PipelineType | str = PipelineType.BASIC,
) -> ComposedPipelineBase:
    """Build a pipeline based on the model path declared in fastvideo_args.

    1. Download the model snapshot if it is not already on disk.
    2. Use registry detectors to look up the pipeline class for that model.
    3. Instantiate and return the pipeline.
    """
    model_path = fastvideo_args.model_path
    model_path = maybe_download_model(model_path)
    logger.info("Model path: %s", model_path)
    logger.info(
        "Building pipeline of type: %s",
        pipeline_type.value if isinstance(pipeline_type, PipelineType) else pipeline_type,
    )

    model_info = get_model_info(
        model_path=model_path,
        pipeline_type=pipeline_type,
        workload_type=fastvideo_args.workload_type,
        override_pipeline_cls_name=fastvideo_args.override_pipeline_cls_name,
    )
    pipeline_cls = model_info.pipeline_cls
    pipeline = pipeline_cls(model_path, fastvideo_args)

    logger.info("Pipelines instantiated")
    return pipeline


__all__ = [
    "ComposedPipelineBase",
    "ForwardBatch",
    "TrainingBatch",
    "build_pipeline",
]
