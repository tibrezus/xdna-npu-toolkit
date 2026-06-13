"""Command-line interface for xdna-npu.

Usage:
    xdna-npu doctor      Full diagnostic + feasibility verdict (the main command)
    xdna-npu detect      NPU hardware detection only
    xdna-npu validate    Stack checks (driver/fw/XRT/plugin/memlock)
    xdna-npu enable      Apply the memlock + power-mode fixes (needs root)
    xdna-npu status      One-line machine-readable status
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .detect import detect
from .enable import enable
from .validate import validate_stack
from .verdict import assess


# --- formatting helpers --------------------------------------------------------

GREEN, RED, YELLOW, BLUE, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[34m", "\033[1m", "\033[0m",
)


def _color_for(status: str) -> str:
    return {PASS: GREEN, FAIL: RED, WARN: YELLOW, INFO: BLUE}.get(status, "")


PASS, FAIL, WARN, INFO = "PASS", "FAIL", "WARN", "INFO"


def _use_color() -> bool:
    return sys.stdout.isatty()


def _c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}" if _use_color() else text


def _box(title: str) -> None:
    print()
    print(_c(f"── {title} ", BOLD) + _c("─" * max(8, 60 - len(title)), BLUE))


# --- commands ------------------------------------------------------------------

def cmd_detect(args: argparse.Namespace) -> int:
    dev = detect()
    if args.json:
        print(json.dumps(_dev_dict(dev), indent=2))
        return 0
    _box("NPU hardware")
    print(f"  PCI address     : {dev.pci_address or 'not found'}")
    print(f"  Vendor:Device   : {_hex(dev.vendor_id)}:{_hex(dev.device_id)}")
    print(f"  Driver          : {dev.driver or '-'}")
    print(f"  Device node     : {dev.device_node or '-'}")
    print(f"  Codename        : {dev.codename or 'unknown'}")
    print(f"  XDNA generation : {dev.xdna_gen or 'unknown'}")
    print(f"  AIE family      : {dev.aie_family or 'unknown'}")
    if dev.aie_info:
        print(f"  AIE array (live): {dev.aie_info.cols} cols x {dev.aie_info.rows} rows, "
              f"col_size={dev.aie_info.col_size}, AIE v{dev.aie_info.aie_version}")
    if dev.firmware_loaded:
        print(f"  Firmware (live) : {dev.firmware_loaded}")
    if dev.firmware_files:
        print(f"  Firmware files  : {len(dev.firmware_files)} in "
              f"/lib/firmware/amdnpu/{_hex(dev.device_id)}_00/")
    if dev.ioctl_error:
        print(_c(f"  ioctl error     : {dev.ioctl_error}", RED))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    dev = detect()
    checks = validate_stack(dev)
    if args.json:
        print(json.dumps([{"name": c.name, "status": c.status.value, "detail": c.detail, "hint": c.hint}
                          for c in checks], indent=2))
        return 0
    _box("Stack validation")
    failed = 0
    for c in checks:
        sym = {PASS: "✓", FAIL: "✗", WARN: "!", INFO: "·"}[c.status.value]
        print(f"  {_c(sym, _color_for(c.status.value))} {_c(f'[{c.status.value}]', _color_for(c.status.value))} {c.name}")
        print(f"      {c.detail}")
        if c.hint:
            print(f"      {_c('→ ' + c.hint, YELLOW)}")
        if c.status.value == FAIL:
            failed += 1
    return 1 if failed else 0


def cmd_doctor(args: argparse.Namespace) -> int:
    rc = cmd_detect(args)
    rc = max(rc, cmd_validate(args))
    dev = detect()
    v = assess(dev)
    _box("LLM feasibility verdict")
    color = GREEN if v.can_run_llm else RED
    print(_c(f"  {v.summary}", color))
    print(f"  {v.detail}")
    if v.recommendation:
        print()
        for line in v.recommendation.splitlines():
            print(f"    {line}")
    print()
    print(_c("─" * 64, BLUE))
    return rc


def cmd_enable(args: argparse.Namespace) -> int:
    _box("Enable NPU (memlock + power mode)")
    res = enable()
    for k, val in res.items():
        if k.endswith("_error"):
            print(_c(f"  ✗ {k}: {val}", RED))
        else:
            print(_c(f"  ✓ {k}: {val}", GREEN))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    dev = detect()
    v = assess(dev)
    cols = dev.aie_info.cols if dev.aie_info else "?"
    print(f"npu={dev.device_node or 'none'} gen={dev.xdna_gen or '?'} cols={cols} "
          f"fw={dev.firmware_loaded or '?'} llm_capable={'yes' if v.can_run_llm else 'no'}")
    return 0


# --- helpers -------------------------------------------------------------------

def _hex(v) -> str:
    return f"0x{v:04x}" if v is not None else "-"


def _dev_dict(dev) -> dict:
    return {
        "pci_address": dev.pci_address,
        "vendor_id": dev.vendor_id,
        "device_id": dev.device_id,
        "driver": dev.driver,
        "device_node": dev.device_node,
        "codename": dev.codename,
        "xdna_gen": dev.xdna_gen,
        "aie_family": dev.aie_family,
        "aie_cols": dev.aie_info.cols if dev.aie_info else None,
        "aie_rows": dev.aie_info.rows if dev.aie_info else None,
        "aie_version": dev.aie_info.aie_version if dev.aie_info else None,
        "firmware_loaded": str(dev.firmware_loaded) if dev.firmware_loaded else None,
        "ioctl_error": dev.ioctl_error,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="xdna-npu",
        description="Detect, validate, enable and assess the AMD XDNA NPU on Linux.",
    )
    p.add_argument("-V", "--version", action="version", version=f"xdna-npu {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--json", action="store_true", help="machine-readable JSON output")

    sp = sub.add_parser("doctor", help="full diagnostic + feasibility verdict")
    add_common(sp)
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("detect", help="NPU hardware detection")
    add_common(sp)
    sp.set_defaults(func=cmd_detect)

    sp = sub.add_parser("validate", help="stack checks (driver/fw/XRT/memlock)")
    add_common(sp)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("enable", help="apply memlock + power-mode fixes (needs root)")
    sp.set_defaults(func=cmd_enable)

    sp = sub.add_parser("status", help="one-line machine-readable status")
    sp.set_defaults(func=cmd_status)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
