# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the SSD spill transport layer (fastvideo.training.ssd_spill).

CPU-only: exercises the zero-copy byte view, aligned staging allocation, and the
direct-I/O file round-trip. On Linux these run over real O_DIRECT; on macOS over
F_NOCACHE. The direct-I/O alignment *contract* (offset, length, and buffer
address) is enforced by the module itself whenever ``direct=True``, so those
tests run on any platform.
"""
from __future__ import annotations

import os

import pytest
import torch

from fastvideo.training import ssd_spill
from fastvideo.training.ssd_spill import (DIRECT_IO_ALIGN, SpillFile,
                                          alloc_aligned_host, align_up,
                                          direct_io_supported, is_aligned,
                                          tensor_byte_view)


def test_align_up() -> None:
    assert align_up(0) == 0
    assert align_up(1) == DIRECT_IO_ALIGN
    assert align_up(DIRECT_IO_ALIGN) == DIRECT_IO_ALIGN
    assert align_up(DIRECT_IO_ALIGN + 1) == 2 * DIRECT_IO_ALIGN


def test_byte_view_is_zero_copy_and_bf16_safe() -> None:
    # bf16 is the SAC payload and has no numpy dtype; the view must still work
    # and must alias the tensor's storage (same data_ptr).
    t = torch.arange(6, dtype=torch.bfloat16)
    mv = tensor_byte_view(t)
    assert mv.nbytes == t.numel() * t.element_size()  # 6 * 2
    assert mv.contiguous and not mv.readonly
    assert t.view(torch.uint8).numpy().__array_interface__["data"][0] \
        == t.data_ptr()
    # Mutating through the view is visible in the tensor -> genuinely zero-copy.
    mv[0] = 0xFF
    assert t.view(torch.uint8)[0].item() == 0xFF


def test_byte_view_rejects_noncontiguous() -> None:
    strided = torch.arange(12, dtype=torch.bfloat16).view(3, 4)[:, ::2]
    assert not strided.is_contiguous()
    with pytest.raises(ValueError):
        tensor_byte_view(strided)


def test_alloc_aligned_host_is_aligned_and_padded() -> None:
    buf = alloc_aligned_host(6)  # bf16 payload of 6 bytes
    assert is_aligned(buf.data_ptr())
    assert buf.numel() == align_up(6) == DIRECT_IO_ALIGN
    assert buf.dtype == torch.uint8


def _stage(t: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Copy ``t`` into an aligned, padded staging buffer; return (buf, nbytes)."""
    nbytes = t.numel() * t.element_size()
    buf = alloc_aligned_host(nbytes)
    tensor_byte_view(buf)[:nbytes] = tensor_byte_view(t)
    return buf, nbytes


def test_spillfile_bf16_roundtrip_byte_exact(tmp_path) -> None:
    t = torch.randn(1024, dtype=torch.float32).bfloat16()
    src, nbytes = _stage(t)
    with SpillFile(str(tmp_path / "spill.bin")) as f:
        f.write(0, src)
        dst = alloc_aligned_host(nbytes)
        f.read_into(0, dst)
    got = dst[:nbytes].view(torch.bfloat16)
    assert torch.equal(got, t)


def test_spillfile_multiple_aligned_regions(tmp_path) -> None:
    a = torch.randn(500).bfloat16()  # deliberately non-block-multiple length
    b = torch.randn(500).bfloat16()
    sa, na = _stage(a)
    sb, nb = _stage(b)
    off_b = align_up(na)
    with SpillFile(str(tmp_path / "multi.bin")) as f:
        f.write(0, sa)
        f.write(off_b, sb)
        da, db = alloc_aligned_host(na), alloc_aligned_host(nb)
        f.read_into(off_b, db)  # out-of-order read
        f.read_into(0, da)
    assert torch.equal(da[:na].view(torch.bfloat16), a)
    assert torch.equal(db[:nb].view(torch.bfloat16), b)


def test_spillfile_loops_over_short_transfers(tmp_path, monkeypatch) -> None:
    """A buffer larger than one syscall chunk still round-trips byte-exactly.

    Linux caps a single pwrite/preadv at 0x7ffff000 bytes and returns a short
    count for anything bigger. Shrinking the chunk exercises the same loop
    without allocating >2 GiB: a regression here reappears in production only
    on generic saved tensors (SAC's ~1.3 GB buffers stay under the real cap).
    """
    monkeypatch.setattr(ssd_spill, "_MAX_IO_CHUNK", DIRECT_IO_ALIGN)
    t = torch.randn(8 * DIRECT_IO_ALIGN, dtype=torch.float32).bfloat16()
    src, nbytes = _stage(t)
    assert nbytes > DIRECT_IO_ALIGN  # more than one chunk, so the loop runs
    with SpillFile(str(tmp_path / "chunked.bin")) as f:
        f.write(0, src)
        dst = alloc_aligned_host(nbytes)
        f.read_into(0, dst)
    assert torch.equal(dst[:nbytes].view(torch.bfloat16), t)


def test_max_io_chunk_is_alignment_safe() -> None:
    # Every chunk boundary must stay valid for O_DIRECT (offset, length, and
    # buffer address all block-aligned), so the cap itself must be a multiple.
    assert ssd_spill._MAX_IO_CHUNK % DIRECT_IO_ALIGN == 0
    assert 0 < ssd_spill._MAX_IO_CHUNK <= 0x7ffff000


def test_direct_rejects_misaligned_buffer_address(tmp_path) -> None:
    # A view starting one byte into an aligned buffer is address-misaligned;
    # the transport must reject it BEFORE os.pwrite (else Linux O_DIRECT EINVAL).
    aligned = alloc_aligned_host(2 * DIRECT_IO_ALIGN)
    misaligned = aligned[1:1 + DIRECT_IO_ALIGN]
    assert not is_aligned(misaligned.data_ptr())
    with SpillFile(str(tmp_path / "addr.bin")) as f:
        with pytest.raises(ValueError, match="address"):
            f.write(0, misaligned)


def test_direct_rejects_misaligned_offset_and_length(tmp_path) -> None:
    buf = alloc_aligned_host(DIRECT_IO_ALIGN)
    with SpillFile(str(tmp_path / "ol.bin")) as f:
        with pytest.raises(ValueError, match="offset"):
            f.write(1, buf)
        short = buf[:100]  # length not block-aligned
        with pytest.raises(ValueError, match="length"):
            f.write(0, short)


def test_nondirect_allows_unaligned(tmp_path) -> None:
    # With direct=False the alignment contract is not enforced.
    t = torch.arange(100, dtype=torch.uint8)
    with SpillFile(str(tmp_path / "buf.bin"), direct=False) as f:
        f.write(0, t)
        dst = torch.empty(100, dtype=torch.uint8)
        f.read_into(0, dst)
    assert torch.equal(dst, t)


def test_direct_io_supported_on_tmp(tmp_path) -> None:
    assert direct_io_supported(str(tmp_path)) is True
    leftovers = [p for p in os.listdir(tmp_path) if "ssd_spill_probe" in p]
    assert leftovers == []
