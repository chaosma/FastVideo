# SPDX-License-Identifier: Apache-2.0
"""Shared training utilities used by fastvideo.train.

This package previously held a parallel set of training pipelines
(e.g. ``WanTrainingPipeline``, ``WanDistillationPipeline``). Those
have been removed in favor of the YAML-driven trainer in
``fastvideo.train``; only the helper modules that the new trainer
imports remain (``training_utils``, ``activation_checkpoint``,
``checkpointing_utils``, ``trackers``).
"""
