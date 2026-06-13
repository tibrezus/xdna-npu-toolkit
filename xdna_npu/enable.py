"""System-level enablement fixes that require root.

These are the two blockers we hit on a stock Arch install of a 7840HS:

  1. RLIMIT_MEMLOCK is ~8 MiB by default -> XRT mmap of DMA buffers fails
     with EAGAIN (Resource temporarily unavailable). Fix: a drop-in under
     /etc/security/limits.d granting the render group unlimited memlock.
  2. NPU clocks default to 'default' power mode. Fix: xrt-smi --pmode
     performance (runtime-only; resets on power cycle).

This module writes the files and shells out; it re-execs under sudo if not
already root (so `xdna-npu enable` Just Works).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

LIMITS_DROPIN = "/etc/security/limits.d/99-amd-npu.conf"
LIMITS_CONTENT = """# Allow accelerator (AMD Ryzen AI NPU / XRT) runtimes to pin DMA buffers.
# xrt/amdxdna mmap BOs at a high offset; the default 8MiB memlock -> EAGAIN.
# Installed by xdna-npu-toolkit.
@render   soft   memlock   unlimited
@render   hard   memlock   unlimited
@render   soft   nofile    1048576
@render   hard   nofile    1048576
"""


def _ensure_root(action: str) -> None:
    if os.geteuid() == 0:
        return
    print(f"[enable] {action} requires root; re-execing with sudo ...", file=sys.stderr)
    # Preserve the calling user's environment/PATH where it matters.
    os.execvp("sudo", ["sudo", "-E", sys.executable, "-m", "xdna_npu", "enable"])


def install_memlock_limits() -> tuple[str, str]:
    _ensure_root("installing memlock limits drop-in")
    os.makedirs("/etc/security/limits.d", exist_ok=True)
    with open(LIMITS_DROPIN, "w") as f:
        f.write(LIMITS_CONTENT)
    os.chmod(LIMITS_DROPIN, 0o644)
    # Make sure `sudo` itself honours limits.d on this Arch setup.
    pam_sudo = "/etc/pam.d/sudo"
    note = ""
    try:
        with open(pam_sudo) as f:
            content = f.read()
        if "pam_limits.so" not in content:
            with open(pam_sudo, "a") as f:
                f.write("session    required   pam_limits.so\n")
            note = f"(also added pam_limits.so to {pam_sudo})"
    except OSError:
        pass
    return (LIMITS_DROPIN, note)


def set_power_mode(mode: str = "performance") -> str:
    """Set the NPU power mode via xrt-smi. Needs root (the SET_STATE ioctl is privileged)."""
    xrt_smi = shutil.which("xrt-smi")
    if not xrt_smi:
        raise RuntimeError("xrt-smi not found; install the `xrt` package")
    _ensure_root(f"setting NPU power mode to {mode}")
    # Find the device BDF.
    r = subprocess.run([xrt_smi, "examine"], capture_output=True, text=True)
    import re

    m = re.search(r"\[([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])\]", r.stdout)
    bdf = m.group(1) if m else None
    cmd = [xrt_smi, "configure", "--pmode", mode]
    if bdf:
        cmd += ["-d", bdf]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"xrt-smi configure failed: {res.stderr.strip() or res.stdout.strip()}")
    return mode


def enable() -> dict:
    results: dict = {}
    try:
        path, note = install_memlock_limits()
        results["memlock"] = f"installed {path} {note}".strip()
    except Exception as exc:  # noqa: BLE001
        results["memlock_error"] = str(exc)
    try:
        results["power_mode"] = set_power_mode("performance")
    except Exception as exc:  # noqa: BLE001
        results["power_mode_error"] = str(exc)
    results["note"] = (
        "Log out and back in (or start a new login shell) for the memlock "
        "limit to take effect for your user."
    )
    return results
