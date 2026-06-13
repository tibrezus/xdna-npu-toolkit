"""Stack validation: kernel driver, device node permissions, XRT, plugin, memlock.

Each check returns a :class:`Check` so the UI can render pass/fail/warn and
give an actionable hint. Checks are independent -- one failure does not skip
the others.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum

from .detect import NpuDevice

PASS, FAIL, WARN, INFO = "PASS", "FAIL", "WARN", "INFO"


class Status(str, Enum):
    PASS = PASS
    FAIL = FAIL
    WARN = WARN
    INFO = INFO


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    hint: str = ""


def _memlock_limit() -> int:
    """Current RLIMIT_MEMLOCK (soft) in BYTES. Returns -1 if unlimited, -2 if unknown."""
    try:
        import resource

        soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        return -1 if soft == resource.RLIM_INFINITY else int(soft)
    except Exception:
        return -2


def _pacman_installed(pkg: str) -> bool:
    if shutil.which("pacman"):
        return subprocess.run(
            ["pacman", "-Q", pkg], capture_output=True, text=True
        ).returncode == 0
    return False


def _xrt_plugin_present() -> bool:
    return os.path.exists("/usr/lib/libxrt_driver_xdna.so.2")


def validate_stack(dev: NpuDevice) -> list[Check]:
    checks: list[Check] = []

    # 1. kernel driver bound
    if dev.driver == "amdxdna":
        checks.append(Check("kernel driver 'amdxdna' bound", Status.PASS,
                            f"PCI {dev.pci_address} -> amdxdna"))
    elif dev.driver:
        checks.append(Check("kernel driver 'amdxdna' bound", Status.FAIL,
                            f"PCI {dev.pci_address} bound to '{dev.driver}' instead",
                            "Install/boot a kernel with the amdxdna driver (>=6.11, best on 7.x)."))
    else:
        checks.append(Check("kernel driver 'amdxdna' bound", Status.FAIL,
                            "no AMD accelerator found on the PCI bus",
                            "This host does not expose an AMD XDNA NPU."))

    # 2. device node + permissions
    if dev.device_node and os.path.exists(dev.device_node):
        if os.access(dev.device_node, os.R_OK | os.W_OK):
            checks.append(Check("device node accessible", Status.PASS,
                                f"{dev.device_node} (current user has r/w)"))
        else:
            checks.append(Check("device node accessible", Status.FAIL,
                                f"{dev.device_node} present but not r/w for current user",
                                "Add your user to the 'render' group, or fix udev permissions."))
    else:
        checks.append(Check("device node accessible", Status.FAIL,
                            "no /dev/accel/accelN node",
                            "Driver not loaded; check `journalctl -k | grep amdxdna`."))

    # 3. firmware loaded by driver
    if dev.firmware_loaded:
        checks.append(Check("NPU firmware loaded", Status.PASS,
                            f"firmware {dev.firmware_loaded}"))
    elif dev.firmware_files:
        checks.append(Check("NPU firmware loaded", Status.WARN,
                            "firmware files present but driver reports no version",
                            "Firmware may not have downloaded; run the linux-firmware update."))
    else:
        checks.append(Check("NPU firmware loaded", Status.FAIL,
                            "no /lib/firmware/amdnpu/<devid>_00/ and no loaded version",
                            "Install the `linux-firmware` package (provides amdnpu/*)."))

    # 4. XRT userspace runtime
    has_xrt = _pacman_installed("xrt") or bool(shutil.which("xrt-smi"))
    if has_xrt:
        checks.append(Check("XRT userspace runtime", Status.PASS,
                            "xrt / xrt-smi present"))
    else:
        checks.append(Check("XRT userspace runtime", Status.FAIL,
                            "xrt not installed",
                            "Arch: `sudo pacman -S xrt xrt-plugin-amdxdna`."))

    # 5. amdxdna XRT plugin (the XRT <-> amdxdna driver shim)
    if _xrt_plugin_present():
        checks.append(Check("XRT amdxdna plugin", Status.PASS,
                            "/usr/lib/libxrt_driver_xdna.so.2"))
    else:
        checks.append(Check("XRT amdxdna plugin", Status.FAIL,
                            "libxrt_driver_xdna.so.2 missing",
                            "Arch: `sudo pacman -S xrt-plugin-amdxdna`."))

    # 6. memlock limit -- the single most common NPU runtime blocker.
    #    XRT mmaps 64MiB+ DMA buffers; needs effectively unlimited memlock.
    ml = _memlock_limit()  # bytes, or -1 unlimited, -2 unknown
    if ml == -1:
        checks.append(Check("memlock (RLIMIT_MEMLOCK)", Status.PASS, "unlimited"))
    elif ml == -2:
        checks.append(Check("memlock (RLIMIT_MEMLOCK)", Status.WARN,
                            "could not read; XRT mmap may fail with EAGAIN"))
    elif ml >= 256 * 1024 * 1024:  # >=256 MiB: enough for typical workloads
        checks.append(Check("memlock (RLIMIT_MEMLOCK)", Status.PASS,
                            f"{ml // 1024 // 1024} MiB"))
    else:
        checks.append(Check("memlock (RLIMIT_MEMLOCK)", Status.FAIL,
                            f"{ml // 1024 // 1024} MiB ({ml} bytes) -- too low; "
                            "XRT mmap of DMA buffers fails with EAGAIN",
                            "Run `sudo xdna-npu enable` to install /etc/security/limits.d/99-amd-npu.conf, then re-login."))

    return checks
