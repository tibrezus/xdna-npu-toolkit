"""Low-level amdxdna DRM ioctl via ctypes (pure stdlib).

Replicates the `DRM_IOCTL_AMDXDNA_GET_INFO` queries that the kernel
`amdxdna` driver exposes through ``/dev/accel/accelN``. These are
*metadata-only* queries: they do not allocate DMA / pinned buffers, so they
work for any user with read/write access to the device node -- no raised
``memlock`` limit and no root required.

Layouts mirror <drm/amdxdna_accel.h> as of the 7.x kernel uapi:

    enum amdxdna_drm_get_param {
        DRM_AMDXDNA_QUERY_AIE_STATUS,          // 0
        DRM_AMDXDNA_QUERY_AIE_METADATA,        // 1   <- we use this
        DRM_AMDXDNA_QUERY_AIE_VERSION,         // 2
        ...
        DRM_AMDXDNA_QUERY_FIRMWARE_VERSION=8,  //    <- and this
        ...
    };

    struct amdxdna_drm_get_info { u32 param; u32 buffer_size; u64 buffer; };  // 16 bytes

The ioctl command number is::

    _IOWR('d', 0x40 + 7, struct amdxdna_drm_get_info)  # 0xC0106447

where ``0x40 + 7`` comes from ``DRM_COMMAND_BASE + DRM_AMDXDNA_GET_INFO``
and ``7`` is the 0-based index of ``DRM_AMDXDNA_GET_INFO`` in
``enum amdxdna_drm_ioctl_id``.
"""

from __future__ import annotations

import ctypes
import fcntl
import os
from dataclasses import dataclass

# --- ioctl number construction (mirrors <asm-generic/ioctl.h> / <drm/drm.h>) ----
#
# The Linux _IOC layout is:
#     nr   in bits [0, 8)
#     type in bits [8, 16)
#     size in bits [16, 30)
#     dir  in bits [30, 32)
# so _IOC(dir, type, nr, size) = (dir<<30) | (size<<16) | (type<<8) | nr.
_IOC_READ = 2
_IOC_WRITE = 1


def _IOC(direction: int, type_: int, nr: int, size: int) -> int:
    return (
        (direction << 30) | (size << 16) | (type_ << 8) | nr
    )


# DRM_IOCTL_BASE = 'd'; DRM_COMMAND_BASE = 0x40; DRM_AMDXDNA_GET_INFO = 7.
# struct amdxdna_drm_get_info is 16 bytes (u32 + u32 + u64).
DRM_IOCTL_AMDXDNA_GET_INFO = _IOC(_IOC_READ | _IOC_WRITE, ord("d"), 0x40 + 7, 16)

DRM_AMDXDNA_QUERY_AIE_METADATA = 1
DRM_AMDXDNA_QUERY_FIRMWARE_VERSION = 8


# --- ctypes mirrors of the kernel structs --------------------------------------

class _QueryAieTileMeta(ctypes.Structure):
    # u16 row_count, row_start, dma_channel_count, lock_count, event_reg_count;
    # u16 pad[3];  -> 16 bytes
    _fields_ = [
        ("row_count", ctypes.c_uint16),
        ("row_start", ctypes.c_uint16),
        ("dma_channel_count", ctypes.c_uint16),
        ("lock_count", ctypes.c_uint16),
        ("event_reg_count", ctypes.c_uint16),
        ("pad", ctypes.c_uint16 * 3),
    ]


class _QueryAieMeta(ctypes.Structure):
    # u32 col_size; u16 cols; u16 rows;
    # struct amdxdna_drm_query_aie_version { u32 major; u32 minor; } version;
    # _QueryAieTileMeta core, mem, shim;
    _fields_ = [
        ("col_size", ctypes.c_uint32),
        ("cols", ctypes.c_uint16),
        ("rows", ctypes.c_uint16),
        ("v_major", ctypes.c_uint32),
        ("v_minor", ctypes.c_uint32),
        ("core", _QueryAieTileMeta),
        ("mem", _QueryAieTileMeta),
        ("shim", _QueryAieTileMeta),
    ]


class _GetInfo(ctypes.Structure):
    _fields_ = [
        ("param", ctypes.c_uint32),
        ("buffer_size", ctypes.c_uint32),
        ("buffer", ctypes.c_uint64),
    ]


@dataclass
class AieInfo:
    cols: int
    rows: int
    col_size: int
    aie_version_major: int
    aie_version_minor: int

    @property
    def aie_version(self) -> str:
        return f"{self.aie_version_major}.{self.aie_version_minor}"


@dataclass
class FirmwareVersion:
    major: int
    minor: int
    patch: int
    build: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}.{self.build}"


class NpuIoctlError(RuntimeError):
    """Raised when the amdxdna GET_INFO ioctl fails."""


def find_device_node() -> str | None:
    """Return the first world-accessible /dev/accel/accelN node, or None."""
    import glob

    for path in sorted(glob.glob("/dev/accel/accel*")):
        if os.access(path, os.R_OK | os.W_OK):
            return path
    return None


def _query(fd: int, param: int, buf: ctypes.Array, *, node: str) -> None:
    gi = _GetInfo(param=param, buffer_size=ctypes.sizeof(buf), buffer=ctypes.addressof(buf))
    try:
        fcntl.ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, gi)
    except OSError as exc:
        raise NpuIoctlError(
            f"GET_INFO(param={param}) on {node} failed: {exc.strerror or exc} (errno {exc.errno})"
        ) from exc


def query_aie_metadata(node: str | None = None) -> AieInfo:
    """Query live AIE tile-array geometry from the driver.

    Works for any user with access to the device node (no root / no memlock).
    """
    node = node or find_device_node()
    if not node:
        raise NpuIoctlError("no accessible /dev/accel/accelN device node found")
    fd = os.open(node, os.O_RDWR)
    try:
        meta = _QueryAieMeta()
        _query(fd, DRM_AMDXDNA_QUERY_AIE_METADATA, meta, node=node)
        return AieInfo(
            cols=meta.cols,
            rows=meta.rows,
            col_size=meta.col_size,
            aie_version_major=meta.v_major,
            aie_version_minor=meta.v_minor,
        )
    finally:
        os.close(fd)


def query_firmware_version(node: str | None = None) -> FirmwareVersion:
    """Query the loaded NPU firmware version from the driver."""
    node = node or find_device_node()
    if not node:
        raise NpuIoctlError("no accessible /dev/accel/accelN device node found")
    fd = os.open(node, os.O_RDWR)
    try:
        fw = (ctypes.c_uint32 * 4)()
        _query(fd, DRM_AMDXDNA_QUERY_FIRMWARE_VERSION, fw, node=node)
        return FirmwareVersion(int(fw[0]), int(fw[1]), int(fw[2]), int(fw[3]))
    finally:
        os.close(fd)
