"""Hardware detection for the AMD XDNA NPU.

Combines three independent signals so the verdict is robust even when one
fails:

  1. PCI enumeration -- find the AMD processing-accelerator (class 0x118000)
     bound to the ``amdxdna`` driver, and read its vendor/device id.
  2. Live driver geometry -- the GET_INFO AIE-metadata ioctl (cols/rows,
     AIE generation). This is authoritative for *this* chip.
  3. Firmware artifacts -- /lib/firmware/amdnpu/<devid>_00/ plus the loaded
     version from the driver.

The PCI device-id table maps the NPU to its XDNA generation:

    0x1502  Phoenix / Hawk Point   XDNA 1   AIE-ML (AIE2)   5 cols  ~10 TOPS
    0x17f0  Strix Point             XDNA 2   AIE2P           8 cols  ~50 TOPS
    (others detected generically from live columns)
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

from .ioctl import (
    AieInfo,
    FirmwareVersion,
    NpuIoctlError,
    query_aie_metadata,
    query_firmware_version,
)

AMD_VENDOR_ID = 0x1022
ACCEL_PCI_CLASS = 0x118000  # "Signal processing controller" / processing accelerator

# device id -> (codename, xdna generation, aie family, nominal cols, nominal tops)
_DEVICE_TABLE = {
    0x1502: ("Phoenix / Hawk Point", "XDNA 1", "AIE-ML (AIE2)", 5, 10),
    0x17F0: ("Strix Point", "XDNA 2", "AIE2P", 8, 50),
}


@dataclass
class NpuDevice:
    pci_address: str | None          # e.g. 0000:c9:00.1
    vendor_id: int | None
    device_id: int | None
    driver: str | None               # amdxdna
    sysfs_path: str | None
    device_node: str | None          # /dev/accel/accel0
    codename: str | None
    xdna_gen: str | None
    aie_family: str | None
    aie_info: AieInfo | None         # live cols/rows from ioctl
    firmware_loaded: FirmwareVersion | None
    firmware_files: list[str] = field(default_factory=list)
    ioctl_error: str | None = None


def _read_sys(p: str) -> str | None:
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return None


def _scan_amd_accel() -> list[tuple[str, dict]]:
    """All AMD processing-accelerators (class 0x118000) on the PCI bus."""
    base = "/sys/bus/pci/devices"
    out = []
    for addr in sorted(os.listdir(base)):
        cls = _read_sys(f"{base}/{addr}/class")
        if cls is None or int(cls, 16) != ACCEL_PCI_CLASS:
            continue
        ven = _read_sys(f"{base}/{addr}/vendor")
        if ven is None or int(ven, 16) != AMD_VENDOR_ID:
            continue
        drv = None
        drv_link = f"{base}/{addr}/driver"
        if os.path.islink(drv_link):
            drv = os.path.basename(os.readlink(drv_link))
        out.append((addr, {
            "vendor_id": int(ven, 16),
            "device_id": int(_read_sys(f"{base}/{addr}/device") or "0", 16),
            "driver": drv,
            "sysfs_path": f"{base}/{addr}",
        }))
    return out


def _find_accel_pci() -> tuple[str, dict] | None:
    """Find the AMD XDNA NPU on the PCI bus.

    Several AMD blocks share PCI class 0x118000 (e.g. the Sensor Fusion Hub on
    7840HS). The NPU is the one bound to the ``amdxdna`` driver, so prefer it;
    fall back to the first known XDNA device id if none is bound yet.
    """
    devs = _scan_amd_accel()
    for addr, info in devs:
        if info["driver"] == "amdxdna":
            return addr, info
    for addr, info in devs:
        if info["device_id"] in _DEVICE_TABLE:
            return addr, info
    return devs[0] if devs else None


def _find_device_node_for(sysfs: str) -> str | None:
    # /sys/class/accel/accel0 -> ../devices/.../<addr>/accel/accel0
    for link in sorted(glob.glob("/sys/class/accel/accel*")):
        real = os.path.realpath(link)
        if sysfs in real:
            minor = os.path.basename(link)
            devpath = f"/dev/accel/{minor}"
            if os.path.exists(devpath):
                return devpath
    return None


def _firmware_files(device_id: int | None) -> list[str]:
    if device_id is None:
        return []
    devid = f"{device_id:04x}"
    d = f"/lib/firmware/amdnpu/{devid}_00"
    if not os.path.isdir(d):
        return []
    files = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isfile(p) or os.path.islink(p):
            files.append(p)
    return files


def detect() -> NpuDevice:
    found = _find_accel_pci()
    addr, pci = (None, {})
    if found:
        addr, pci = found

    node = None
    if pci.get("sysfs_path"):
        node = _find_device_node_for(pci["sysfs_path"])
    if not node:
        # fall back to any accessible accel node
        for cand in sorted(glob.glob("/dev/accel/accel*")):
            if os.access(cand, os.R_OK | os.W_OK):
                node = cand
                break

    device_id = pci.get("device_id")
    codename = xdna_gen = aie_family = None
    if device_id in _DEVICE_TABLE:
        codename, xdna_gen, aie_family, _, _ = _DEVICE_TABLE[device_id]

    aie: AieInfo | None = None
    fw_loaded: FirmwareVersion | None = None
    ioctl_error = None
    if node:
        try:
            aie = query_aie_metadata(node)
        except NpuIoctlError as exc:
            ioctl_error = str(exc)
        try:
            fw_loaded = query_firmware_version(node)
        except NpuIoctlError:
            pass

    # Refine generation from live columns if PCI id was unknown.
    if aie and xdna_gen is None:
        if aie.cols >= 8:
            xdna_gen, codename, aie_family = "XDNA 2", "Strix-class", "AIE2P"
        else:
            xdna_gen, codename, aie_family = "XDNA 1", "Phoenix-class", "AIE-ML (AIE2)"

    return NpuDevice(
        pci_address=addr,
        vendor_id=pci.get("vendor_id"),
        device_id=pci.get("device_id"),
        driver=pci.get("driver"),
        sysfs_path=pci.get("sysfs_path"),
        device_node=node,
        codename=codename,
        xdna_gen=xdna_gen,
        aie_family=aie_family,
        aie_info=aie,
        firmware_loaded=fw_loaded,
        firmware_files=_firmware_files(device_id),
        ioctl_error=ioctl_error,
    )
