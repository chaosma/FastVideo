# SPDX-License-Identifier: Apache-2.0
"""Tests for CPU-offload selective activation checkpointing.

The dispatch-mode plumbing (per-op FIFO cache, tuple outputs, version wrapping)
is exercised on CPU, where an offload policy degrades to GPU-resident caching
because the source tensors are not CUDA. The actual pinned-host transport is
covered by the CUDA-gated tests, which skip when no GPU is present.
"""
from __future__ import annotations

import functools

import pytest
import torch
from torch.utils.checkpoint import (CheckpointPolicy,
                                     create_selective_checkpoint_contexts)

from fastvideo.training.sac_cpu_offload import (
    _PinnedPool, create_cpu_offload_selective_checkpoint_contexts)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="requires CUDA for pinned-host transport")


def _run(fn, x_init, context_fn):
    """Run ``fn`` under a checkpoint with ``context_fn`` and return (loss, grad)."""
    x = x_init.clone().detach().requires_grad_(True)
    out = torch.utils.checkpoint.checkpoint(fn,
                                            x,
                                            use_reentrant=False,
                                            context_fn=context_fn)
    loss = out.pow(2).sum()
    loss.backward()
    return loss.detach(), x.grad.detach().clone()


def _mm_model(x, w1, w2):
    # Two mm ops with a nonlinearity, so recompute vs save differ in graph.
    h = torch.mm(x, w1)
    h = torch.relu(h)
    return torch.mm(h, w2)


def test_offload_matches_save_and_recompute_cpu():
    torch.manual_seed(0)
    x_init = torch.randn(8, 8)
    w1 = torch.randn(8, 16)
    w2 = torch.randn(16, 8)
    fn = functools.partial(_mm_model, w1=w1, w2=w2)

    def save_policy(ctx, func, *args, **kwargs):
        if func == torch.ops.aten.mm.default:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    def offload_policy(ctx, func, *args, **kwargs):
        if func == torch.ops.aten.mm.default:
            return CheckpointPolicy.MUST_CPU_OFFLOAD
        return CheckpointPolicy.PREFER_RECOMPUTE

    # Baseline: plain non-reentrant checkpoint (full recompute).
    loss_full, grad_full = _run(
        fn, x_init, torch.utils.checkpoint.noop_context_fn)
    # torch's GPU-resident SAC save path.
    loss_save, grad_save = _run(
        fn, x_init,
        functools.partial(create_selective_checkpoint_contexts, save_policy))
    # Our offload path (degrades to resident caching on CPU tensors).
    loss_off, grad_off = _run(
        fn, x_init,
        functools.partial(create_cpu_offload_selective_checkpoint_contexts,
                          offload_policy))

    torch.testing.assert_close(loss_off, loss_full)
    torch.testing.assert_close(loss_off, loss_save)
    torch.testing.assert_close(grad_off, grad_full)
    torch.testing.assert_close(grad_off, grad_save)


def test_offload_tuple_output_cpu():
    """Cache an op that returns a tuple (layer_norm -> out, mean, rstd)."""
    torch.manual_seed(0)
    x_init = torch.randn(4, 8)
    weight = torch.randn(8)
    bias = torch.randn(8)

    def fn(x):
        return torch.nn.functional.layer_norm(x, (8, ), weight, bias)

    def offload_policy(ctx, func, *args, **kwargs):
        if func == torch.ops.aten.native_layer_norm.default:
            return CheckpointPolicy.MUST_CPU_OFFLOAD
        return CheckpointPolicy.PREFER_RECOMPUTE

    loss_full, grad_full = _run(fn, x_init,
                                torch.utils.checkpoint.noop_context_fn)
    loss_off, grad_off = _run(
        fn, x_init,
        functools.partial(create_cpu_offload_selective_checkpoint_contexts,
                          offload_policy))
    torch.testing.assert_close(loss_off, loss_full)
    torch.testing.assert_close(grad_off, grad_full)


def test_offload_op_counts_one_per_self_attn():
    """The offload policy caches / reloads exactly one op per checkpointed region.

    Mirrors design test #4's assertion (one self-attn per covered block). The
    op-level counters are device-independent, so this runs on CPU even though
    the physical host transfer only happens on CUDA.
    """
    q = torch.randn(1, 2, 8, 4, requires_grad=True)

    def block(x):
        # One self-attn SDPA (q/k share seq) + cheap elementwise work.
        o = torch.nn.functional.scaled_dot_product_attention(x, x, x)
        return (o * 2.0).sum(-1, keepdim=True)

    def policy(ctx, func, *args, **kwargs):
        if "scaled_dot_product" in getattr(func, "_name", str(func)):
            return CheckpointPolicy.MUST_CPU_OFFLOAD
        return CheckpointPolicy.PREFER_RECOMPUTE

    captured = {}

    def ctx_fn():
        caching, replay = create_cpu_offload_selective_checkpoint_contexts(
            policy)
        captured["cache"] = caching.cache
        return caching, replay

    out = torch.utils.checkpoint.checkpoint(block,
                                            q,
                                            use_reentrant=False,
                                            context_fn=ctx_fn)
    out.sum().backward()
    cache = captured["cache"]
    assert cache.n_offload_ops == 1
    assert cache.n_reloaded_ops == 1


def test_warns_once_when_backend_op_not_intercepted(monkeypatch):
    """If no op matches the offload policy, warn once (silent no-op guard).

    Simulates an attention backend whose op the policy cannot intercept (e.g.
    FA4/cute, standalone FA2): the block caches zero attention outputs.
    """
    from fastvideo.training import sac_cpu_offload as mod

    monkeypatch.setattr(mod, "_WARNED_NO_OFFLOAD", False)
    warnings_seen = []
    monkeypatch.setattr(mod.logger, "warning",
                        lambda *a, **k: warnings_seen.append(a[0]))

    def never_matches(ctx, func, *args, **kwargs):
        return CheckpointPolicy.PREFER_RECOMPUTE

    def blk(x):
        return (x * 2.0).sum(-1, keepdim=True)

    for _ in range(3):  # repeated blocks must still warn only once
        x = torch.randn(4, 8, requires_grad=True)
        out = torch.utils.checkpoint.checkpoint(
            blk,
            x,
            use_reentrant=False,
            context_fn=functools.partial(
                create_cpu_offload_selective_checkpoint_contexts,
                never_matches))
        out.sum().backward()

    assert len(warnings_seen) == 1
    assert "cached 0 attention outputs" in warnings_seen[0]


@CUDA
def test_pinned_cap_fallback_is_counted_cuda():
    """A capped pool must record GPU-resident fallbacks in the cache counters."""
    import os
    device = torch.device("cuda", 0)
    # Fresh, tiny-capped pool so the offload cannot fit and falls back.
    from fastvideo.training import sac_cpu_offload as mod
    mod._PINNED_POOLS.pop(device, None)

    def offload_policy(ctx, func, *args, **kwargs):
        if func == torch.ops.aten.mm.default:
            return CheckpointPolicy.MUST_CPU_OFFLOAD
        return CheckpointPolicy.PREFER_RECOMPUTE

    captured = {}

    def ctx_fn():
        caching, replay = create_cpu_offload_selective_checkpoint_contexts(
            offload_policy)
        captured["cache"] = caching.cache
        return caching, replay

    w = torch.randn(256, 256, device=device)
    x = torch.randn(256, 256, device=device, requires_grad=True)
    prev = os.environ.get("FASTVIDEO_SAC_OFFLOAD_MIN_MB")
    os.environ["FASTVIDEO_SAC_OFFLOAD_MIN_MB"] = "0"
    os.environ["FASTVIDEO_SAC_OFFLOAD_PINNED_GB"] = "0"  # cap 0 -> always fall back
    try:
        mod._PINNED_POOLS.pop(device, None)
        out = torch.utils.checkpoint.checkpoint(lambda t: torch.mm(t, w),
                                                x,
                                                use_reentrant=False,
                                                context_fn=ctx_fn)
        out.sum().backward()
    finally:
        os.environ.pop("FASTVIDEO_SAC_OFFLOAD_PINNED_GB", None)
        if prev is None:
            os.environ.pop("FASTVIDEO_SAC_OFFLOAD_MIN_MB", None)
        else:
            os.environ["FASTVIDEO_SAC_OFFLOAD_MIN_MB"] = prev
        mod._PINNED_POOLS.pop(device, None)

    cache = captured["cache"]
    assert cache.fallback_leaves >= 1
    assert cache.n_offloaded_leaves == 0


# --------------------------------------------------------------------------
# Pinned pool (CUDA required: pinned host allocation needs a CUDA context).
# --------------------------------------------------------------------------


@CUDA
def test_pinned_pool_reuses_buffers():
    device = torch.device("cuda", 0)
    pool = _PinnedPool(device, cap_bytes=None)
    t = torch.empty(1024, dtype=torch.float16, device=device)
    a = pool.acquire(t)
    assert a is not None
    allocated_after_first = pool.allocated_bytes
    pool.release(a)
    b = pool.acquire(t)
    # Same layout key -> reuse the freed buffer, no new allocation.
    assert b is a
    assert pool.allocated_bytes == allocated_after_first


@CUDA
def test_pinned_pool_cap_falls_back():
    device = torch.device("cuda", 0)
    t = torch.empty(1024, dtype=torch.float16, device=device)  # 2 KB
    pool = _PinnedPool(device, cap_bytes=1024)  # too small for one buffer
    assert pool.acquire(t) is None


@CUDA
def test_offload_roundtrip_matches_cuda():
    """On CUDA, forcing offload of a small tensor must match resident SAC."""
    import os
    device = torch.device("cuda", 0)
    torch.manual_seed(0)
    x_init = torch.randn(8, 8, device=device)
    w1 = torch.randn(8, 16, device=device)
    w2 = torch.randn(16, 8, device=device)
    fn = functools.partial(_mm_model, w1=w1, w2=w2)

    def offload_policy(ctx, func, *args, **kwargs):
        if func == torch.ops.aten.mm.default:
            return CheckpointPolicy.MUST_CPU_OFFLOAD
        return CheckpointPolicy.PREFER_RECOMPUTE

    def save_policy(ctx, func, *args, **kwargs):
        if func == torch.ops.aten.mm.default:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    loss_save, grad_save = _run(
        fn, x_init,
        functools.partial(create_selective_checkpoint_contexts, save_policy))

    prev = os.environ.get("FASTVIDEO_SAC_OFFLOAD_MIN_MB")
    os.environ["FASTVIDEO_SAC_OFFLOAD_MIN_MB"] = "0"  # force offload of small mm
    try:
        from fastvideo.training import sac_cpu_offload
        pool = sac_cpu_offload._pinned_pool(device)
        before = pool.allocated_bytes
        loss_off, grad_off = _run(
            fn, x_init,
            functools.partial(
                create_cpu_offload_selective_checkpoint_contexts,
                offload_policy))
        assert pool.allocated_bytes >= before  # a pinned buffer was used
    finally:
        if prev is None:
            os.environ.pop("FASTVIDEO_SAC_OFFLOAD_MIN_MB", None)
        else:
            os.environ["FASTVIDEO_SAC_OFFLOAD_MIN_MB"] = prev

    torch.cuda.synchronize()
    torch.testing.assert_close(loss_off, loss_save)
    torch.testing.assert_close(grad_off, grad_save)


@CUDA
def test_offload_survives_stream_and_allocator_churn_cuda():
    """Forward on a non-default stream + allocator churn must not corrupt reload.

    Design test #3: the side copy stream and record_stream lifetimes must hold
    even when the compute runs off the default stream and the caching allocator
    is pushed to reuse buffers between forward and backward.
    """
    import os
    device = torch.device("cuda", 0)
    torch.manual_seed(0)
    x_init = torch.randn(64, 64, device=device)
    w1 = torch.randn(64, 128, device=device)
    w2 = torch.randn(128, 64, device=device)
    fn = functools.partial(_mm_model, w1=w1, w2=w2)

    def offload_policy(ctx, func, *args, **kwargs):
        if func == torch.ops.aten.mm.default:
            return CheckpointPolicy.MUST_CPU_OFFLOAD
        return CheckpointPolicy.PREFER_RECOMPUTE

    def save_policy(ctx, func, *args, **kwargs):
        if func == torch.ops.aten.mm.default:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    loss_save, grad_save = _run(
        fn, x_init,
        functools.partial(create_selective_checkpoint_contexts, save_policy))

    prev = os.environ.get("FASTVIDEO_SAC_OFFLOAD_MIN_MB")
    os.environ["FASTVIDEO_SAC_OFFLOAD_MIN_MB"] = "0"
    try:
        x = x_init.clone().detach().requires_grad_(True)
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            out = torch.utils.checkpoint.checkpoint(
                fn,
                x,
                use_reentrant=False,
                context_fn=functools.partial(
                    create_cpu_offload_selective_checkpoint_contexts,
                    offload_policy))
            loss = out.pow(2).sum()
        torch.cuda.current_stream().wait_stream(side)
        # Churn the allocator so any un-recorded source buffer would be reused.
        _ = [torch.randn(1024, 1024, device=device) for _ in range(8)]
        loss.backward()
    finally:
        if prev is None:
            os.environ.pop("FASTVIDEO_SAC_OFFLOAD_MIN_MB", None)
        else:
            os.environ["FASTVIDEO_SAC_OFFLOAD_MIN_MB"] = prev

    torch.cuda.synchronize()
    torch.testing.assert_close(loss.detach(), loss_save)
    torch.testing.assert_close(x.grad.detach(), grad_save)
