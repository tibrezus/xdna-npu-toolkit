"""Minimal, hardware-independent tests for the VitisAI probe."""

import os
import stat
import subprocess
import sys
import tempfile
from unittest import mock

from xdna_npu import embed


def test_find_python312_consistent():
    p = embed._find_python312()
    if p is None:
        return
    r = subprocess.run([p, "--version"], capture_output=True, text=True, timeout=10)
    assert "3.12" in (r.stdout + r.stderr)


def test_site_packages_of_current():
    sp = embed.site_packages_of(sys.executable)
    assert sp and os.path.isdir(sp)


def test_ep_env_shape():
    env = embed.ep_env(sys.executable)
    assert "voe" + os.sep + "lib" in env["LD_LIBRARY_PATH"]
    assert "ryzenai_dynamic_dispatch" in env["DD_ROOT"]


def _fake_site(tmp):
    site = os.path.join(tmp, "site")
    os.makedirs(os.path.join(site, "voe"))
    return site


def test_probe_timeout_handled(tmp_path):
    with mock.patch.object(embed, "site_packages_of", return_value=_fake_site(tmp_path)), \
         mock.patch.object(embed.subprocess, "run",
                           side_effect=subprocess.TimeoutExpired(cmd="x", timeout=240)):
        p = embed.probe_ep(python="python3.12")
    assert not p.session_initialized
    assert any("timed out" in m for m in p.messages)


def test_probe_parses_stub_and_cleans_tmp(tmp_path):
    site = _fake_site(tmp_path)
    stub = tmp_path / "stub.sh"
    stub.write_text("#!/bin/sh\n"
                    "echo '__AVPS__VitisAIExecutionProvider,CPUExecutionProvider'\n"
                    "echo '__SESS__VitisAIExecutionProvider'\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    before = set(os.listdir(tempfile.gettempdir()))
    with mock.patch.object(embed, "site_packages_of", return_value=site):
        p = embed.probe_ep(python=str(stub))
    assert p.vitisai_available and p.session_initialized
    assert "xdna_tiny.onnx" not in os.listdir(tempfile.gettempdir())
    assert not (set(os.listdir(tempfile.gettempdir())) - before)
