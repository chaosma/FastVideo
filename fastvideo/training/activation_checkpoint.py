import collections
import logging
import os
from enum import Enum

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, offload_wrapper)

from fastvideo.training.activation_streaming import (
    StreamedOffloadCheckpointWrapper)

logger = logging.getLogger(__name__)

TRANSFORMER_BLOCK_NAMES = [
    "blocks",
    "double_blocks",
    "single_blocks",
    "transformer_blocks",
    "temporal_transformer_blocks",
    "transformer_double_blocks",
    "transformer_single_blocks",
]


class CheckpointType(str, Enum):
    FULL = "full"
    OPS = "ops"
    BLOCK_SKIP = "block_skip"
    # Naive offload baseline -- uses PyTorch's offload_wrapper, so H2D / D2H
    # runs on the compute stream (no overlap).
    FULL_OFFLOAD = "full_offload"
    # Phase 2 streamed-offload mode -- H2D / D2H on a dedicated copy stream.
    STREAMED_OFFLOAD = "streamed_offload"


_SELECTIVE_ACTIVATION_CHECKPOINTING_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
}


def apply_activation_checkpointing(module: torch.nn.Module,
                                   checkpointing_type: str = CheckpointType.FULL,
                                   n_layer: int = 1) -> torch.nn.Module:
    if checkpointing_type == CheckpointType.FULL:
        module = _apply_activation_checkpointing_blocks(module)
    elif checkpointing_type == CheckpointType.OPS:
        module = _apply_activation_checkpointing_ops(module, _SELECTIVE_ACTIVATION_CHECKPOINTING_OPS)
    elif checkpointing_type == CheckpointType.BLOCK_SKIP:
        module = _apply_activation_checkpointing_blocks(module, n_layer)
    elif checkpointing_type == CheckpointType.FULL_OFFLOAD:
        module = _apply_activation_checkpointing_blocks(
            module, wrapper=_wrap_offload)
    elif checkpointing_type == CheckpointType.STREAMED_OFFLOAD:
        effective = _maybe_fallback_streamed_to_full_offload()
        if effective == CheckpointType.STREAMED_OFFLOAD:
            module = _apply_activation_checkpointing_blocks(
                module, wrapper=_wrap_streamed_offload)
        else:
            module = _apply_activation_checkpointing_blocks(
                module, wrapper=_wrap_offload)
    else:
        raise ValueError(
            f"Checkpointing type '{checkpointing_type}' not supported. Supported types are {CheckpointType.__members__.keys()}"
        )
    return module


def _wrap_recompute(block: torch.nn.Module) -> torch.nn.Module:
    return checkpoint_wrapper(block, preserve_rng_state=False)


def _wrap_offload(block: torch.nn.Module) -> torch.nn.Module:
    return offload_wrapper(checkpoint_wrapper(block, preserve_rng_state=False))


def _wrap_streamed_offload(block: torch.nn.Module) -> torch.nn.Module:
    return StreamedOffloadCheckpointWrapper(
        checkpoint_wrapper(block, preserve_rng_state=False))


# Default per-local-rank pinned-host budget required by STREAMED_OFFLOAD on
# the 14B target (≈68 GB stash + slack). On 5B this is well under the
# default. Override via ``FASTVIDEO_STREAMED_OFFLOAD_MIN_HOST_GB``.
_DEFAULT_STREAMED_HOST_GB_PER_RANK = 80.0


def _read_meminfo_available_gb() -> float | None:
    """Return host MemAvailable in GiB (Linux only). None if unavailable."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = float(line.split()[1])
                    return kb / (1024 * 1024)
    except (FileNotFoundError, ValueError, IndexError):
        return None
    return None


def _maybe_fallback_streamed_to_full_offload() -> "CheckpointType":
    """Verify enough pinned host memory is available for STREAMED_OFFLOAD;
    fall back to FULL_OFFLOAD with a warning otherwise.

    Disabled entirely by setting ``FASTVIDEO_STREAMED_OFFLOAD_SKIP_HOST_CHECK=1``.
    The threshold can be tuned with ``FASTVIDEO_STREAMED_OFFLOAD_MIN_HOST_GB``
    (per local rank, in GiB).
    """
    if os.environ.get("FASTVIDEO_STREAMED_OFFLOAD_SKIP_HOST_CHECK") == "1":
        return CheckpointType.STREAMED_OFFLOAD

    avail_gb = _read_meminfo_available_gb()
    if avail_gb is None:
        # Non-Linux or unreadable /proc -- don't second-guess the user.
        return CheckpointType.STREAMED_OFFLOAD

    local_world_size = int(
        os.environ.get("LOCAL_WORLD_SIZE")
        or os.environ.get("NPROC_PER_NODE") or 1)
    per_rank_gb = avail_gb / max(local_world_size, 1)

    threshold = float(
        os.environ.get("FASTVIDEO_STREAMED_OFFLOAD_MIN_HOST_GB")
        or _DEFAULT_STREAMED_HOST_GB_PER_RANK)

    if per_rank_gb >= threshold:
        logger.info(
            "STREAMED_OFFLOAD host-RAM check OK: %.1f GiB available / "
            "%d local ranks = %.1f GiB per rank (threshold %.1f GiB).",
            avail_gb, local_world_size, per_rank_gb, threshold)
        return CheckpointType.STREAMED_OFFLOAD

    logger.warning(
        "STREAMED_OFFLOAD requires ~%.1f GiB of pinned host memory per "
        "local rank, but only %.1f GiB is available (%.1f GiB total / "
        "%d local ranks). Falling back to FULL_OFFLOAD. Set "
        "FASTVIDEO_STREAMED_OFFLOAD_SKIP_HOST_CHECK=1 to override, or "
        "FASTVIDEO_STREAMED_OFFLOAD_MIN_HOST_GB=<value> to retune.",
        threshold, per_rank_gb, avail_gb, local_world_size)
    return CheckpointType.FULL_OFFLOAD


def _apply_activation_checkpointing_blocks(
        module: torch.nn.Module,
        n_layer: int | None = None,
        wrapper=_wrap_recompute) -> torch.nn.Module:
    applied = False
    for transformer_block_name in TRANSFORMER_BLOCK_NAMES:
        blocks: torch.nn.Module = getattr(module, transformer_block_name, None)
        if blocks is None:
            continue
        for index, (layer_id, block) in enumerate(blocks.named_children()):
            if n_layer is None or index % n_layer == 0:
                block = wrapper(block)
                blocks.register_module(layer_id, block)
        applied = True
    if not applied:
        raise ValueError("Activation checkpointing is not applied successfully")
    return module


def _apply_activation_checkpointing_ops(module: torch.nn.Module, ops) -> torch.nn.Module:
    from torch.utils.checkpoint import (CheckpointPolicy, create_selective_checkpoint_contexts)

    def _get_custom_policy(meta: dict[str, int]) -> CheckpointPolicy:

        def _custom_policy(ctx, func, *args, **kwargs):
            mode = "recompute" if ctx.is_recompute else "forward"
            mm_count_key = f"{mode}_mm_count"
            if func == torch.ops.aten.mm.default:
                meta[mm_count_key] += 1
            # Saves output of all compute ops, except every second mm
            to_save = func in ops and not (func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0)
            return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE

        return _custom_policy

    def selective_checkpointing_context_fn():
        meta: dict[str, int] = collections.defaultdict(int)
        return create_selective_checkpoint_contexts(_get_custom_policy(meta))

    return checkpoint_wrapper(module, context_fn=selective_checkpointing_context_fn, preserve_rng_state=False)
