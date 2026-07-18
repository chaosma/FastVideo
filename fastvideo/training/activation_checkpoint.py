import collections
from enum import Enum

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper)

from fastvideo.logger import init_logger

logger = init_logger(__name__)

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
    SELF_ATTN = "self_attn"
    # Same as SELF_ATTN, but the saved self-attn outputs are streamed to pinned
    # CPU memory during forward and back to GPU during recompute, trading the
    # GPU-resident SAC cache floor for PCIe traffic. See sac_cpu_offload.py.
    SELF_ATTN_CPU = "self_attn_cpu"


_SELECTIVE_ACTIVATION_CHECKPOINTING_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
}

# Scaled-dot-product-attention aten ops, both with (query, key, value, ...) as
# their leading positional args. These carry q/k/v as [B, num_heads, seq, dim]
# (seq is dim -2). Used by the self-attn save policy.
_ATTENTION_OPS = {
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
}

# FlashAttention-3 (the FLASH_ATTN backend when ``flash_attn_interface`` is
# installed) runs attention through ``torch.ops.flash_attn_3.fwd`` instead of
# the aten SDPA op. Unlike the standalone FA2 ``flash_attn`` op, this FA3 op IS
# visible to the selective-checkpoint TorchDispatch policy, so we can MUST_SAVE
# the self-attention output under FA3 too. FA3 carries q/k/v as
# [B, seq, num_heads, dim] (seq is dim 1) -- a different layout from aten SDPA.
_FA3_ATTENTION_OPS: set = set()
try:
    import flash_attn_interface  # noqa: F401  (loads the flash_attn_3 op library)
    _FA3_ATTENTION_OPS = {torch.ops.flash_attn_3.fwd.default}
except Exception:
    logger.debug(
        "flash_attn_interface (FA3) not importable; the self-attn save policy "
        "cannot intercept FA3 attention ops (_FA3_ATTENTION_OPS is empty).")


def apply_activation_checkpointing(module: torch.nn.Module,
                                   checkpointing_type:
                                   str = CheckpointType.FULL,
                                   n_layer: int = 1) -> torch.nn.Module:
    base, coverage = _parse_checkpointing_type(checkpointing_type)
    if base == CheckpointType.FULL:
        module = _apply_activation_checkpointing_blocks(module)
    elif base == CheckpointType.OPS:
        _warn_if_attention_not_interceptable()
        module = _apply_activation_checkpointing_ops(
            module, _SELECTIVE_ACTIVATION_CHECKPOINTING_OPS)
    elif base == CheckpointType.BLOCK_SKIP:
        module = _apply_activation_checkpointing_blocks(module, n_layer)
    elif base == CheckpointType.SELF_ATTN:
        _warn_if_attention_not_interceptable()
        module = _apply_self_attn_checkpointing(module, coverage)
    elif base == CheckpointType.SELF_ATTN_CPU:
        _warn_if_attention_not_interceptable()
        module = _apply_self_attn_checkpointing(
            module, coverage, context_fn=_self_attn_cpu_offload_context_fn)
    else:
        raise ValueError(
            f"Checkpointing type '{checkpointing_type}' not supported. "
            f"Supported types are {CheckpointType.__members__.keys()}"
        )
    return module


def _parse_checkpointing_type(
        checkpointing_type: str | CheckpointType) -> tuple[str, float]:
    """Return the base checkpoint type and optional coverage fraction.

    String configs may carry a coverage suffix, e.g. ``self_attn:0.5``. Enum
    inputs need normalization through ``.value`` because
    ``str(CheckpointType.FULL)`` is ``"CheckpointType.FULL"``, not ``"full"``.
    """
    type_str = (checkpointing_type.value
                if isinstance(checkpointing_type, CheckpointType) else
                str(checkpointing_type))
    base, _, coverage_str = type_str.partition(":")
    return base, float(coverage_str) if coverage_str else 1.0


def _apply_activation_checkpointing_blocks(
        module: torch.nn.Module,
        n_layer: int | None = None) -> torch.nn.Module:
    applied = False
    for transformer_block_name in TRANSFORMER_BLOCK_NAMES:
        blocks: torch.nn.Module = getattr(module, transformer_block_name, None)
        if blocks is None:
            continue
        for index, (layer_id, block) in enumerate(blocks.named_children()):
            if n_layer is None or index % n_layer == 0:
                block = checkpoint_wrapper(block, preserve_rng_state=False)
                blocks.register_module(layer_id, block)
        applied = True
    if not applied:
        raise ValueError("Activation checkpointing is not applied successfully")
    return module


def _apply_activation_checkpointing_ops(module: torch.nn.Module,
                                        ops) -> torch.nn.Module:
    from torch.utils.checkpoint import (CheckpointPolicy,
                                        create_selective_checkpoint_contexts)

    def _get_custom_policy(meta: dict[str, int]) -> CheckpointPolicy:

        def _custom_policy(ctx, func, *args, **kwargs):
            mode = "recompute" if ctx.is_recompute else "forward"
            mm_count_key = f"{mode}_mm_count"
            if func == torch.ops.aten.mm.default:
                meta[mm_count_key] += 1
            # Saves output of all compute ops, except every second mm
            to_save = func in ops and not (
                func == torch.ops.aten.mm.default
                and meta[mm_count_key] % 2 == 0)
            return (CheckpointPolicy.MUST_SAVE if to_save else
                    CheckpointPolicy.PREFER_RECOMPUTE)

        return _custom_policy

    def selective_checkpointing_context_fn():
        meta: dict[str, int] = collections.defaultdict(int)
        return create_selective_checkpoint_contexts(_get_custom_policy(meta))

    return checkpoint_wrapper(module,
                              context_fn=selective_checkpointing_context_fn,
                              preserve_rng_state=False)


def _is_self_attention(args: tuple) -> bool:
    """True for self-attention SDPA calls (query/key share a sequence length).

    Cross-attention attends video queries against the short text K/V, so its
    key sequence length differs from the query's; self-attention's matches.
    """
    if len(args) < 2:
        return False
    query, key = args[0], args[1]
    if not (isinstance(query, torch.Tensor) and isinstance(key, torch.Tensor)):
        return False
    return query.shape[-2] == key.shape[-2]


def _is_self_attention_fa3(args: tuple) -> bool:
    """True for self-attention FA3 calls (``flash_attn_3.fwd``).

    FA3 lays q/k/v out as [B, seq, num_heads, dim], so the sequence length is
    dim 1 (not -2 as in the aten SDPA ops). Self-attention shares the query and
    key sequence length; cross-attention's short text K/V differs.
    """
    if len(args) < 2:
        return False
    query, key = args[0], args[1]
    if not (isinstance(query, torch.Tensor) and isinstance(key, torch.Tensor)):
        return False
    if query.dim() < 2 or key.dim() < 2:
        return False
    return query.shape[1] == key.shape[1]


def _warn_if_attention_not_interceptable() -> None:
    """Warn if attention won't dispatch through the interceptable aten SDPA op.

    The self-attn / ops save policy matches ``aten._scaled_dot_product_*``,
    which only runs when attention goes through
    ``F.scaled_dot_product_attention`` (the TORCH_SDPA backend). With the
    standalone ``flash_attn`` package, attention is a custom op the policy
    cannot see, so it gets silently recomputed and the speedup is lost.
    """
    # The FLASH_ATTN backend picks its kernel in this order (see
    # attention/backends/flash_attn.py): FA4 (``flash_attn.cute``) > FA3
    # (``flash_attn_interface``) > FA2 (standalone ``flash_attn``). Of these,
    # only FA3 dispatches through an op the save policy can intercept
    # (``flash_attn_3.fwd``; see _FA3_ATTENTION_OPS) -- FA4's cute op and the
    # FA2 op are both invisible to it. So the warning can only be suppressed
    # when FA3 is the version that will actually run, i.e. FA4 is not
    # importable but FA3 is; otherwise fall through and warn.
    def _fa4_available() -> bool:
        try:
            from fastvideo.attention.utils.flash_attn_cute import (  # noqa: F401
                flash_attn_func)
            return True
        except Exception:
            return False

    if _FA3_ATTENTION_OPS and not _fa4_available():
        return
    try:
        from fastvideo.attention.selector import (
            get_env_variable_attn_backend, get_global_forced_attn_backend)
        from fastvideo.platforms import AttentionBackendEnum
        from fastvideo.utils import STR_BACKEND_ENV_VAR
    except Exception:
        return
    forced = get_global_forced_attn_backend() or get_env_variable_attn_backend()
    if forced == AttentionBackendEnum.TORCH_SDPA:
        return
    if forced != AttentionBackendEnum.FLASH_ATTN:
        # Auto-selection: SDPA is used only when flash_attn is unavailable.
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            logger.info(
                "SAC: flash_attn not installed; attention uses torch SDPA "
                "(interceptable by the save policy).")
            return
    logger.warning(
        "SAC attention-save is enabled but attention will likely use the "
        "standalone flash_attn backend, whose op the policy CANNOT intercept "
        "-- attention will be recomputed and the speedup lost. Set "
        "%s=TORCH_SDPA to force the interceptable SDPA path.",
        STR_BACKEND_ENV_VAR)


def _self_attn_sac_context_fn():
    from torch.utils.checkpoint import (CheckpointPolicy,
                                        create_selective_checkpoint_contexts)

    def _policy(ctx, func, *args, **kwargs):
        # Save the self-attention output (skips the expensive flash forward
        # recompute in backward); recompute everything else, including the
        # cheap cross-attention.
        if func in _ATTENTION_OPS and _is_self_attention(args):
            return CheckpointPolicy.MUST_SAVE
        if func in _FA3_ATTENTION_OPS and _is_self_attention_fa3(args):
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(_policy)


def _self_attn_cpu_offload_context_fn():
    from torch.utils.checkpoint import CheckpointPolicy

    from fastvideo.training.sac_cpu_offload import (
        create_cpu_offload_selective_checkpoint_contexts)

    def _policy(ctx, func, *args, **kwargs):
        # Offload the self-attention output to pinned CPU (reloaded during
        # recompute, skipping the flash forward); recompute everything else.
        if func in _ATTENTION_OPS and _is_self_attention(args):
            return CheckpointPolicy.MUST_CPU_OFFLOAD
        if func in _FA3_ATTENTION_OPS and _is_self_attention_fa3(args):
            return CheckpointPolicy.MUST_CPU_OFFLOAD
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_cpu_offload_selective_checkpoint_contexts(_policy)


def _covered_indices(n_blocks: int, coverage: float) -> set[int]:
    """Pick round(coverage * n_blocks) block indices, spread over [0, n)."""
    m = max(0, min(n_blocks, round(coverage * n_blocks)))
    return {
        i
        for i in range(n_blocks)
        if (i + 1) * m // n_blocks > i * m // n_blocks
    }


def _apply_self_attn_checkpointing(
        module: torch.nn.Module,
        coverage: float = 1.0,
        context_fn=_self_attn_sac_context_fn) -> torch.nn.Module:
    applied = False
    for transformer_block_name in TRANSFORMER_BLOCK_NAMES:
        blocks: torch.nn.Module = getattr(module, transformer_block_name, None)
        if blocks is None:
            continue
        children = list(blocks.named_children())
        covered = _covered_indices(len(children), coverage)
        for index, (layer_id, block) in enumerate(children):
            if index in covered:
                # Save self-attn output, recompute the rest of the block.
                block = checkpoint_wrapper(
                    block,
                    context_fn=context_fn,
                    preserve_rng_state=False)
            else:
                # Uncovered blocks use plain activation checkpointing: only the
                # block inputs are saved, and every activation (self-attn
                # included) is recomputed in backward. Covered blocks above
                # additionally keep the self-attn output so its expensive flash
                # forward is not recomputed.
                block = checkpoint_wrapper(block, preserve_rng_state=False)
            blocks.register_module(layer_id, block)
        applied = True
    if not applied:
        raise ValueError("Activation checkpointing is not applied successfully")
    return module
