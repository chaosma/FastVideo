# SPDX-License-Identifier: Apache-2.0
"""SSD spill transport: zero-copy, direct-I/O tensor <-> file byte movement.

Foundational byte-transport layer for the SSD residency tier that sits beneath
SAC CPU offload (see ``.agents/exploration/ssd-offload-design.md``). This module
owns *only* the transport: opening a direct-I/O spill file and moving a dense
CPU tensor's bytes to/from an aligned file region with no intermediate Python
copy. Residency policy, budgeting, and scheduling live in higher layers.

Direct I/O (``O_DIRECT`` on Linux) is required so spilled bytes bypass the
kernel page cache -- otherwise the RAM we are trying to evict stays resident in
cache and the feature saves nothing. On non-Linux dev boxes (macOS) ``O_DIRECT``
does not exist; we fall back to ``F_NOCACHE``, which is advisory but lets the
transport be exercised in tests. Buffered I/O is never used on Linux.
"""
from __future__ import annotations

import ctypes
import os
import sys

import torch

from fastvideo.logger import init_logger

logger = init_logger(__name__)

# Direct I/O requires the buffer address, file offset, and transfer length to be
# aligned to the device logical block size. 4096 covers every common NVMe/page
# size and is exactly what a pinned (page-aligned) host allocation already
# satisfies for the buffer address.
DIRECT_IO_ALIGN = 4096

# Linux caps a single read/write syscall at 0x7ffff000 bytes (~2 GiB) and
# returns a short count for anything larger -- it is not an error. Generic
# saved tensors routinely exceed that at 4K (a 14B norm_out save is ~2.4 GiB),
# whereas SAC's per-block buffers (~1.3 GB) never do, so only the generic path
# ever tripped it. Transfers are issued in chunks of this size and looped until
# the whole buffer moves. Kept alignment-sized so every chunk boundary stays
# valid for O_DIRECT (offset, length, and buffer address must all be aligned).
_MAX_IO_CHUNK = 0x7ffff000 - (0x7ffff000 % DIRECT_IO_ALIGN)

_IS_LINUX = sys.platform.startswith("linux")
_IS_DARWIN = sys.platform == "darwin"
# macOS fcntl command to disable the buffer cache for a fd (advisory O_DIRECT
# analogue). Value from <sys/fcntl.h>; not exposed by the fcntl module.
_F_NOCACHE = 48


def align_up(n: int, block: int = DIRECT_IO_ALIGN) -> int:
    """Round ``n`` up to a multiple of ``block`` (the direct-I/O quantum)."""
    return (n + block - 1) // block * block


def tensor_byte_view(t: torch.Tensor) -> memoryview:
    """Return a writable, contiguous, zero-copy byte view over ``t``'s storage.

    bf16 (the main SAC payload) has no NumPy dtype, and both
    ``memoryview(tensor)`` and ``memoryview(tensor.untyped_storage())`` raise, so
    the direct buffer-protocol routes do not work. We reinterpret to ``uint8``
    first -- ``t.view(torch.uint8)`` is a zero-copy view sharing ``data_ptr``,
    and ``uint8`` *is* a NumPy dtype, so ``.numpy()`` does not copy. A ``ctypes``
    view over ``data_ptr()`` is the fallback (works for any dtype as raw bytes).

    The caller MUST keep ``t`` alive for the lifetime of the returned view: both
    paths alias ``t``'s storage without owning it (the ctypes path in
    particular holds only a raw address).
    """
    if t.device.type != "cpu":
        raise ValueError("tensor_byte_view requires a CPU tensor")
    if not t.is_contiguous():
        raise ValueError("tensor_byte_view requires a contiguous tensor")
    nbytes = t.numel() * t.element_size()
    try:
        arr = t.view(torch.uint8).numpy()
        if arr.__array_interface__["data"][0] == t.data_ptr():
            mv = memoryview(arr)
            if mv.nbytes == nbytes and mv.contiguous:
                return mv
    except Exception:
        pass
    # Fallback: raw ctypes view over the data pointer, as bytes.
    buf = (ctypes.c_char * nbytes).from_address(t.data_ptr())
    return memoryview(buf)


def is_aligned(ptr: int, block: int = DIRECT_IO_ALIGN) -> bool:
    return ptr % block == 0


def alloc_aligned_host(nbytes: int, *, pin: bool = False) -> torch.Tensor:
    """Allocate a block-aligned, block-padded ``uint8`` CPU staging buffer.

    Direct I/O requires the buffer *address* -- not just offset/length -- to be
    block-aligned. Pinned (page-aligned) allocations already satisfy this, so the
    fast path allocates exactly ``align_up(nbytes)`` bytes -- no over-allocation,
    so the caller's RSS accounting (which charges ``padded``) is exact. Ordinary
    CPU tensors are not reliably aligned (observed alternating alignment; a bare
    ``bytearray`` is consistently 2048 mod 4096), so only when the exact
    allocation lands misaligned do we over-allocate one block and return an
    aligned view (keeping its backing storage alive).
    """
    padded = align_up(nbytes)
    buf = torch.empty(padded, dtype=torch.uint8, pin_memory=pin)
    if is_aligned(buf.data_ptr()):
        return buf
    raw = torch.empty(padded + DIRECT_IO_ALIGN, dtype=torch.uint8,
                      pin_memory=pin)
    off = (-raw.data_ptr()) % DIRECT_IO_ALIGN
    buf = raw[off:off + padded]
    if not is_aligned(buf.data_ptr()):
        raise RuntimeError("aligned host allocation failed")
    return buf


def _open_direct(path: str, flags: int) -> int:
    """Open ``path`` for direct (page-cache-bypassing) I/O and return the fd."""
    if _IS_LINUX:
        flags |= getattr(os, "O_DIRECT", 0)
    fd = os.open(path, flags, 0o600)
    if _IS_DARWIN:
        import fcntl
        fcntl.fcntl(fd, _F_NOCACHE, 1)
    return fd


class SpillFile:
    """A direct-I/O backing file for spilled tensor bytes.

    Regions are addressed by aligned byte offset. The caller owns offset
    allocation and buffer padding; this class validates alignment and moves
    bytes zero-copy. The file is unlinked on open (anonymous), so it is
    reclaimed automatically when the fd closes or the process dies.
    """

    def __init__(self, path: str, *, direct: bool = True) -> None:
        self.path = path
        # ``direct`` requests page-cache bypass; honored via O_DIRECT (Linux) or
        # F_NOCACHE (macOS dev).
        self._direct = direct
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        if direct:
            self.fd = _open_direct(path, flags)
        else:
            self.fd = os.open(path, flags, 0o600)
        # Unlink immediately: the fd keeps the storage live, but the name is
        # gone, so a crash cannot leak the spill file.
        os.unlink(path)

    def _validate(self, offset: int, buf: torch.Tensor) -> int:
        if buf.device.type != "cpu" or not buf.is_contiguous():
            raise ValueError("spill buffer must be a contiguous CPU tensor")
        nbytes = buf.numel() * buf.element_size()
        # Direct I/O requires the file offset, transfer length, AND buffer
        # address to be block-aligned. Enforce whenever direct was requested --
        # not only on Linux -- so the macOS/F_NOCACHE dev path catches a
        # violation of the invariant before it reaches a real O_DIRECT fd (where
        # it would surface as an opaque EINVAL). Callers get aligned, padded
        # buffers from ``alloc_aligned_host``.
        if self._direct:
            if offset % DIRECT_IO_ALIGN:
                raise ValueError(
                    f"direct I/O offset {offset} not {DIRECT_IO_ALIGN}-aligned")
            if nbytes % DIRECT_IO_ALIGN:
                raise ValueError(
                    f"direct I/O length {nbytes} not {DIRECT_IO_ALIGN}-aligned; "
                    "allocate with alloc_aligned_host")
            if not is_aligned(buf.data_ptr()):
                raise ValueError(
                    f"direct I/O buffer address not {DIRECT_IO_ALIGN}-aligned; "
                    "allocate with alloc_aligned_host")
        return nbytes

    def write(self, offset: int, buf: torch.Tensor) -> None:
        """Write the whole padded ``buf`` at ``offset`` (aligned staging tensor).

        Loops over short writes: a single ``pwrite`` transfers at most
        ``_MAX_IO_CHUNK`` bytes, and a partial transfer is normal POSIX
        behaviour rather than a failure.
        """
        nbytes = self._validate(offset, buf)
        view = tensor_byte_view(buf)
        done = 0
        while done < nbytes:
            chunk = min(_MAX_IO_CHUNK, nbytes - done)
            n = os.pwrite(self.fd, view[done:done + chunk], offset + done)
            if n <= 0:
                raise OSError(
                    f"short spill write: {done + max(n, 0)}/{nbytes} bytes")
            done += n

    def read_into(self, offset: int, buf: torch.Tensor) -> None:
        """Read the whole padded ``buf`` at ``offset`` into it (readinto).

        Loops over short reads for the same reason ``write`` does; a short
        count here means the syscall cap, not EOF, because the spill file
        always holds the full padded extent that ``write`` put there.
        """
        nbytes = self._validate(offset, buf)
        view = tensor_byte_view(buf)
        done = 0
        while done < nbytes:
            chunk = min(_MAX_IO_CHUNK, nbytes - done)
            n = os.preadv(self.fd, [view[done:done + chunk]], offset + done)
            if n <= 0:
                raise OSError(
                    f"short spill read: {done + max(n, 0)}/{nbytes} bytes")
            done += n

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> SpillFile:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


def direct_io_supported(directory: str) -> bool:
    """Probe whether direct I/O actually works on ``directory``'s filesystem.

    Rented instances can present overlay/network mounts as "NVMe"; direct I/O
    typically fails to open or round-trip on those. Returns True only if an
    aligned direct write+read round-trips byte-exactly. Never raises.
    """
    probe = os.path.join(directory, f".ssd_spill_probe.{os.getpid()}")
    try:
        os.makedirs(directory, exist_ok=True)
        # Use the same aligned-buffer path as runtime spill I/O, so a failure
        # means the filesystem/device rejects direct I/O -- not a buffer
        # alignment bug in the probe itself.
        src = alloc_aligned_host(4)
        tensor_byte_view(src)[:4] = memoryview(b"FVSD")
        dst = alloc_aligned_host(4)
        with SpillFile(probe, direct=True) as f:
            f.write(0, src)
            f.read_into(0, dst)
        return bytes(tensor_byte_view(dst)[:4]) == b"FVSD"
    except Exception as e:
        logger.warning("SSD spill: direct-I/O probe failed on %s (%s); the "
                       "path may be overlay/network-backed, not local NVMe.",
                       directory, e)
        return False
