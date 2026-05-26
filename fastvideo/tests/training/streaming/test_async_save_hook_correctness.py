"""Correctness tests for the AsyncCpuSaveHook in activation_streaming.

These tests verify two things:

1. The pack/unpack pair is bit-exact across a variety of dtypes and shapes
   (round-trip equality after H2D / D2H on the copy stream).
2. End-to-end autograd through a simple module wrapped in
   StreamedOffloadCheckpointWrapper produces gradients identical to running
   the same module without the hook (the hook must be transparent).

The integration test that compares STREAMED vs FULL_OFFLOAD in a real
training step lives separately.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="streaming hook requires CUDA")

from fastvideo.training.activation_streaming import (  # noqa: E402
    AsyncCpuSaveHook,
    BlockRegistry,
    StreamedOffloadCheckpointWrapper,
    get_copy_stream,
)


@pytest.fixture(autouse=True)
def _fresh_registry():
    # Each test installs its own AsyncCpuSaveHook + BlockRegistry, so module
    # singletons don't interfere. The module singletons (get_copy_stream,
    # get_registry) are exercised in the wrapper integration test below.
    yield


@pytest.mark.parametrize("dtype", [
    torch.float32,
    torch.bfloat16,
    torch.float16,
])
@pytest.mark.parametrize("shape", [
    (1, ),
    (17, 31),
    (2, 3, 4, 5),
])
def test_pack_unpack_roundtrip_bit_exact(dtype, shape):
    """pack(t); unpack(packed) must reproduce t bit-exactly."""
    device = torch.device("cuda")
    stream = get_copy_stream(device)
    registry = BlockRegistry(stream)
    registry.new_call(block_idx=0)
    hook = AsyncCpuSaveHook(registry, block_idx=0)

    # Generate a tensor on the compute stream.
    torch.manual_seed(0)
    t = torch.randn(shape, dtype=torch.float32, device=device).to(dtype)

    with hook:
        packed = hook.pack_hook(t)
    restored = hook.unpack_hook(packed)

    # Synchronize so the bit-comparison sees the final copies.
    torch.cuda.synchronize(device)

    assert restored.device == t.device
    assert restored.dtype == t.dtype
    assert restored.shape == t.shape
    # bit-exact -- H2D / D2H is a memcpy, no precision loss.
    assert torch.equal(restored, t), \
        f"pack/unpack roundtrip not bit-exact for dtype={dtype}, shape={shape}"


def test_pack_passes_through_cpu_tensors():
    """Non-CUDA tensors are returned verbatim from pack; unpack mirrors that."""
    stream = get_copy_stream(torch.device("cuda"))
    registry = BlockRegistry(stream)
    registry.new_call(block_idx=0)
    hook = AsyncCpuSaveHook(registry, block_idx=0)

    cpu_t = torch.randn(4, 5)
    packed = hook.pack_hook(cpu_t)
    assert packed is cpu_t  # not staged via H2D
    restored = hook.unpack_hook(packed)
    assert restored is cpu_t


def test_pack_passes_through_empty_tensors():
    """Zero-size tensors are returned verbatim -- not worth the bookkeeping."""
    stream = get_copy_stream(torch.device("cuda"))
    registry = BlockRegistry(stream)
    registry.new_call(block_idx=0)
    hook = AsyncCpuSaveHook(registry, block_idx=0)

    empty = torch.empty(0, device="cuda", dtype=torch.bfloat16)
    packed = hook.pack_hook(empty)
    assert packed is empty


def test_multiple_packs_distinct_slots():
    """Packing several tensors in one block call must produce distinct slots,
    and the registry must hold them in order."""
    stream = get_copy_stream(torch.device("cuda"))
    registry = BlockRegistry(stream)
    registry.new_call(block_idx=0)
    hook = AsyncCpuSaveHook(registry, block_idx=0)

    tensors = [torch.randn(8, device="cuda") + i for i in range(5)]
    packs = [hook.pack_hook(t) for t in tensors]

    slots = [p[2] for p in packs]
    assert slots == list(range(5))
    assert len(registry.records[0]) == 5

    restored = [hook.unpack_hook(p) for p in packs]
    torch.cuda.synchronize()
    for t, r in zip(tensors, restored):
        assert torch.equal(t, r)


class _TinyMLP(torch.nn.Module):
    """Two-layer MLP used as a stand-in transformer block for the hook
    transparency check."""

    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(32, 64, bias=True)
        self.lin2 = torch.nn.Linear(64, 32, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(torch.nn.functional.gelu(self.lin1(x)))


def _clone_params(mod: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in mod.parameters()]


def test_wrapper_forward_backward_matches_unwrapped():
    """Forward output and parameter gradients must match between the plain
    module and the StreamedOffloadCheckpointWrapper version.

    The wrapper composes with the standard recompute checkpoint to mirror the
    production composition (StreamedOffloadCheckpointWrapper(
    checkpoint_wrapper(block))), but for this transparency test we use a
    bare wrapper -- if the bare hook is non-transparent, composition with
    checkpoint_wrapper won't fix it.
    """
    torch.manual_seed(123)
    device = torch.device("cuda")
    plain = _TinyMLP().to(device).to(torch.float32)
    streamed = _TinyMLP().to(device).to(torch.float32)
    # Same weights.
    streamed.load_state_dict(plain.state_dict())

    wrapped = StreamedOffloadCheckpointWrapper(streamed)

    x = torch.randn(8, 32, device=device, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)

    y_plain = plain(x)
    y_wrap = wrapped(x2)

    torch.cuda.synchronize()
    assert torch.allclose(y_plain, y_wrap, atol=0, rtol=0), \
        "wrapped forward output diverges from unwrapped"

    loss_plain = y_plain.pow(2).sum()
    loss_wrap = y_wrap.pow(2).sum()
    loss_plain.backward()
    loss_wrap.backward()
    torch.cuda.synchronize()

    for (n, p_p), (_, p_w) in zip(plain.named_parameters(),
                                  streamed.named_parameters()):
        assert p_p.grad is not None and p_w.grad is not None, n
        assert torch.allclose(p_p.grad, p_w.grad, atol=0, rtol=0), \
            f"grad mismatch on {n}"

    assert x.grad is not None and x2.grad is not None
    assert torch.allclose(x.grad, x2.grad, atol=0, rtol=0)
