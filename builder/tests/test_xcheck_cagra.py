"""Cross-check (black box) of Phase 5 step 4, the cuVS scaffold, builder side.

Spec only: Phase 5 decision 10 ("without cuVS importable the CLI refuses in
one line (exit 1), which is what every laptop and kind run sees"), the step
2–4 notes ("build --builder cagra, which fails fast with a reason where cuVS
is not importable, before any fetch"), Phase 4 decision 17 (the CLI's
contract: one JSON document on stdout, exit 1 carries a failure document),
`build --help`, and the docstrings of kgpu_builder.cagra. The caller's
brief: {"status":"failed","reason":"CuvsUnavailable: ...","build_id":...,
"attempt":...}, before touching the store, under a few seconds.

"Before touching the store" is observed with a TCP listener standing in for
S3_ENDPOINT that counts the connections it is offered.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from kgpu_builder import cagra

BUILDER_DIR = Path(__file__).resolve().parents[1]


class CountingListener:
    """A stand-in object store: accepts TCP connections, counts them, closes them."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(64)
        self.sock.settimeout(0.2)
        self.port = self.sock.getsockname()[1]
        self.count = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                c, _ = self.sock.accept()
            except (socket.timeout, OSError):
                continue
            self.count += 1
            c.close()

    def close(self) -> None:
        self._stop.set()
        self._t.join(2)
        self.sock.close()


@pytest.fixture
def listener():
    lst = CountingListener()
    yield lst
    lst.close()


def env_for(endpoint: str) -> dict[str, str]:
    e = {k: v for k, v in os.environ.items() if not k.startswith("S3_")}
    e.update(S3_ENDPOINT=endpoint, S3_ACCESS_KEY="x", S3_SECRET_KEY="y", S3_BUCKET="xcheck-none", S3_REGION="us-east-1")
    return e


def run_build(endpoint: str, extra: list[str], build_id: str = "4:999", attempt: int = 2, timeout: float = 180):
    argv = [sys.executable, "-m", "kgpu_builder", "build", "--build-id", build_id, "--attempt", str(attempt),
            "--dataset", "sift-128", "--n-shards", "6", "--shard", "4", "--threads", "2"] + extra
    t0 = time.monotonic()
    p = subprocess.run(argv, capture_output=True, text=True, env=env_for(endpoint), cwd=BUILDER_DIR, timeout=timeout)
    dt = time.monotonic() - t0
    print(f"ran: build ... {' '.join(extra) or '(no --builder)'} with S3_ENDPOINT={endpoint} -> exit {p.returncode} in {dt:.2f}s")
    print(f"     stdout: {p.stdout.strip()[:400]!r}")
    print(f"     stderr (last 300): {p.stderr.strip()[-300:]!r}")
    return p, dt


def python_startup_s() -> float:
    """The cost of starting the interpreter and importing the package, for scale."""
    t0 = time.monotonic()
    subprocess.run([sys.executable, "-c", "import kgpu_builder.cli"], cwd=BUILDER_DIR, capture_output=True)
    return time.monotonic() - t0


# --------------------------------------------------------------------------


def test_xcheck_cagra_help_lists_builder():
    p = subprocess.run([sys.executable, "-m", "kgpu_builder", "build", "--help"], capture_output=True, text=True, cwd=BUILDER_DIR)
    line = next((l.strip() for l in p.stdout.splitlines() if "--builder" in l), "")
    print(f"build --help exit {p.returncode}; the --builder usage: {line!r}")
    print("spec: brief / decision 10: build --help lists --builder {hnswlib,cagra}")
    assert p.returncode == 0
    assert "--builder {hnswlib,cagra}" in p.stdout


def test_xcheck_cagra_module_surface_without_cuvs():
    print("spec: kgpu_builder.cagra docstring: where cuVS is not importable cuvs_available() is False and build_index_cagra raises CuvsUnavailable with one line")
    avail, ver = cagra.cuvs_available(), cagra.cuvs_version()
    print(f"cuvs_available() = {avail!r}; cuvs_version() = {ver!r}; CuvsUnavailable mro = {[c.__name__ for c in cagra.CuvsUnavailable.__mro__]}")
    try:
        import cuvs  # noqa: F401
        pytest.skip("cuVS is importable here; this test is for the laptop")
    except ImportError:
        pass
    assert avail is False
    assert isinstance(ver, str) and ver
    assert issubclass(cagra.CuvsUnavailable, Exception)
    vecs = np.random.default_rng(1).standard_normal((50, 8)).astype(np.float32)
    seen = []
    with pytest.raises(cagra.CuvsUnavailable) as ei:
        cagra.build_index_cagra(vecs, 16, 200, 2, seen.append)
    msg = str(ei.value)
    print(f"build_index_cagra(50x8) raised CuvsUnavailable({msg!r}); progress calls before raising: {seen}")
    assert "\n" not in msg.strip(), "one line"
    assert msg.strip(), "a reason"


def test_xcheck_cagra_cli_refuses_before_the_store(listener):
    endpoint = f"http://127.0.0.1:{listener.port}"
    base = python_startup_s()
    print(f"(for scale: interpreter + import kgpu_builder.cli takes {base:.2f}s)")
    p, dt = run_build(endpoint, ["--builder", "cagra"], build_id="4:999", attempt=2)
    print(f"connections offered to the stand-in store: {listener.count}")
    print("spec: decision 10 + notes: exit 1, one line, a reason, before any fetch; brief: "
          "{\"status\":\"failed\",\"reason\":\"CuvsUnavailable: ...\",\"build_id\":...,\"attempt\":...}, under a few seconds")
    assert p.returncode == 1
    lines = [l for l in p.stdout.splitlines() if l.strip()]
    assert len(lines) == 1, f"stdout must be one line, got {len(lines)}"
    doc = json.loads(lines[0])
    print(f"failure document: {doc}")
    assert doc.get("status") == "failed"
    assert str(doc.get("reason", "")).startswith("CuvsUnavailable: "), doc.get("reason")
    assert doc.get("build_id") == "4:999"
    assert doc.get("attempt") == 2 and isinstance(doc.get("attempt"), int)
    assert listener.count == 0, "the cagra refusal must not touch the store"
    assert dt < 5.0, f"took {dt:.2f}s; the refusal should be fast"
    for l in p.stderr.splitlines():
        if l.strip():
            try:
                json.loads(l)
            except ValueError:
                pytest.fail(f"stderr line is not JSON (decision 17: progress and phases as JSON lines): {l!r}")


def test_xcheck_cagra_cli_refuses_with_a_closed_port():
    # a port nobody listens on: the refusal does not depend on what the store does
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    p, dt = run_build(f"http://127.0.0.1:{port}", ["--builder", "cagra"], build_id="0:1", attempt=1)
    doc = json.loads(p.stdout)
    print(f"closed port {port}: exit {p.returncode} in {dt:.2f}s, reason {doc.get('reason')!r}")
    assert p.returncode == 1 and doc["reason"].startswith("CuvsUnavailable: ") and dt < 5.0


@pytest.mark.parametrize("extra", [[], ["--builder", "hnswlib"]], ids=["default", "hnswlib"])
def test_xcheck_hnswlib_still_goes_to_the_store(listener, extra):
    endpoint = f"http://127.0.0.1:{listener.port}"
    p, dt = run_build(endpoint, extra, build_id="4:999", attempt=2)
    print(f"connections offered to the stand-in store: {listener.count}")
    print("spec: brief: --builder hnswlib and the default behave as before: fetch from the store first (Phase 4 decision 17), "
          "so against a store that hangs up the build fails with exit 1 and one failure document, not with CuvsUnavailable")
    assert p.returncode == 1
    doc = json.loads(p.stdout)
    print(f"failure document: {doc}")
    assert doc.get("status") == "failed"
    assert not str(doc.get("reason", "")).startswith("CuvsUnavailable"), "hnswlib must not need cuVS"
    assert listener.count > 0, "the hnswlib path must reach the store (it fetches the shard)"


# ----------------------------------------------------- with a real store (scripts/test-db.sh --py)

S3 = os.environ.get("TEST_S3_ENDPOINT")


@pytest.mark.skipif(not S3, reason="TEST_S3_ENDPOINT unset")
def test_xcheck_cagra_real_store_writes_nothing_and_hnswlib_builds(tmp_path):
    from test_xcheck_build import Toy  # the Phase 4 cross-check's toy split in a fresh bucket

    toy = Toy(tmp_path, n_shards=2, size=2000, dim=16)
    try:
        pc = toy.run("1:5", 3, shard=1, extra=["--builder", "cagra"])
        print(f"cagra: exit {pc.returncode}, stdout {pc.stdout.strip()!r}; artifacts after: {toy.artifacts()}")
        assert pc.returncode == 1
        assert json.loads(pc.stdout)["reason"].startswith("CuvsUnavailable: ")
        assert toy.artifacts() == [], "the refusal writes nothing under artifacts/"
        for label, extra, bid in [("default", [], "1:6"), ("hnswlib", ["--builder", "hnswlib"], "1:7")]:
            p = toy.run(bid, 1, shard=1, extra=extra)
            m = json.loads(p.stdout)
            print(f"{label}: exit {p.returncode}; manifest kind={m.get('kind')!r} cuvs_version={m.get('cuvs_version')!r} "
                  f"hnswlib_version={m.get('hnswlib_version')!r} recall={m.get('recall')}; artifacts {toy.artifacts('artifacts/' + bid + '/')}")
            assert p.returncode == 0
            assert m.get("build_id") == bid and m.get("attempt") == 1
            assert m.get("kind") in (None, "hnswlib"), f"{label}: an hnswlib build must not claim kind {m.get('kind')!r}"
            assert toy.artifacts(f"artifacts/{bid}/1/") == [f"artifacts/{bid}/1/index.bin", f"artifacts/{bid}/1/manifest.json"]
    finally:
        toy.close()


@pytest.mark.parametrize("bad", [["--threads", "0"], ["--build-id", "a/b"]], ids=["threads0", "slash"])
def test_xcheck_cagra_usage_errors_still_exit_2(bad):
    argv = [sys.executable, "-m", "kgpu_builder", "build", "--build-id", "1:1", "--attempt", "1", "--dataset", "sift-128",
            "--n-shards", "6", "--shard", "0", "--builder", "cagra"] + bad
    p = subprocess.run(argv, capture_output=True, text=True, env=env_for("http://127.0.0.1:1"), cwd=BUILDER_DIR, timeout=60)
    print(f"--builder cagra {' '.join(bad)} -> exit {p.returncode}; stderr {p.stderr.strip()!r}; stdout {p.stdout.strip()!r}")
    print("spec: Phase 4 decision 17: a usage error exits 2 with one line on stderr; decision 10 adds the cagra refusal as an exit 1, it does not say it overrides usage")
    assert p.returncode == 2
    assert len([l for l in p.stderr.splitlines() if l.strip()]) == 1
