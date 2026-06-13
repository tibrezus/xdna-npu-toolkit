#!/usr/bin/env bash
# Standalone enablement script for hosts without Python, or for systemd/early-boot use.
# Applies the two blockers that keep a stock XDNA NPU from being usable:
#   1. memlock drop-in so XRT can mmap DMA buffers (no more EAGAIN)
#   2. NPU power mode -> performance (full clocks)
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "needs root; re-running with sudo" >&2
  exec sudo -E bash "$0" "$@"
fi

echo "[1/2] Installing memlock limits drop-in for the 'render' group ..."
mkdir -p /etc/security/limits.d
cat > /etc/security/limits.d/99-amd-npu.conf <<'EOF'
# Allow accelerator (AMD Ryzen AI NPU / XRT) runtimes to pin DMA buffers.
# xrt/amdxdna mmap BOs at a high offset; the default 8MiB memlock -> EAGAIN.
@render   soft   memlock   unlimited
@render   hard   memlock   unlimited
@render   soft   nofile    1048576
@render   hard   nofile    1048576
EOF
if [ -f /etc/pam.d/sudo ] && ! grep -q pam_limits /etc/pam.d/sudo; then
  echo "session    required   pam_limits.so" >> /etc/pam.d/sudo
fi
echo "    -> /etc/security/limits.d/99-amd-npu.conf installed"

echo "[2/2] Setting NPU power mode to performance ..."
if command -v xrt-smi >/dev/null 2>&1; then
  BDF=$(xrt-smi examine 2>/dev/null | grep -oE '\[[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]\]' | tr -d '[]' | head -1)
  if [ -n "$BDF" ]; then
    xrt-smi configure -d "$BDF" --pmode performance || echo "    -> WARN: could not set power mode (non-fatal)"
  fi
else
  echo "    -> xrt-smi not found; install the 'xrt' package to set power mode"
fi

echo
echo "Done. Log out and back in (or open a fresh login shell) for the memlock"
echo "limit to take effect for your user."
