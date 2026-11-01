"""Cross-check (black box) of Phase 5 step 4, the cuVS scaffold, worker side.

Spec only: Phase 5 decision 10 ("The builder is selected per job by
--builder cagra on the CLI and BUILDER=cagra on the worker, the same
child-process path as hnswlib (Phase 4 decision 18)"; the GPU image on
rapidsai/base:26.10-cuda12-py3.12 with the builder and the worker installed on
top, worker/Dockerfile.gpu; the CPU image unchanged, decision 1), the step 2–4
notes ("BUILDER=cagra: the same child process with --builder cagra
(invariant 9); the Artifact's kind from the manifest"), the HnswlibBuilder
docstring ("With builder='cagra' ... the Artifact's kind is the manifest's
'cagra-hnsw'; nothing else differs"), the ConfigError docstring ("__main__
prints it and exits 2") and the caller's brief (by_name("cagra", ...) returns
a builder named "cagra-hnsw"; `--builder cagra` at the END of the argv).

The builder CLI is replaced by stub scripts that record their argv; the
Dockerfile is checked as text only (never built here: a multi-GB base).
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

import pytest

from kgpu_worker.builder import Artifact, Cancelled, GateRefused, Job, by_name
from kgpu_worker.config import Config, ConfigError
from kgpu_worker.costmodel import Model

WORKER_DIR = Path(__file__).resolve().parents[1]
REPO = WORKER_DIR.parent

STUB = r'''
import json, os, sys, time
d = os.environ["XC_DIR"]
args = sys.argv[1:]
with open(os.path.join(d, "argv.%d" % os.getpid()), "w") as f:
    json.dump(args, f)
mode = os.environ.get("XC_MODE", "ok")
bid = args[args.index("--build-id") + 1]; att = int(args[args.index("--attempt") + 1])
for p in (0.25, 0.5, 1.0):
    print(json.dumps({"progress": p}), file=sys.stderr, flush=True)
if mode == "hang":
    time.sleep(60)
if mode == "gate":
    print(json.dumps({"status": "failed", "reason": "recall@10 0.50 < 0.90"})); sys.exit(3)
if mode == "fail":
    print(json.dumps({"status": "failed", "reason": "Boom: one line", "build_id": bid, "attempt": att})); sys.exit(1)
m = {"status": "done", "build_id": bid, "attempt": att, "n_vectors": 1000, "dim": 128, "index_bytes": 4242,
     "index_sha256": "00", "recall": 0.99, "build_s": 0.1}
kind = os.environ.get("XC_KIND", "")
if kind != "-":
    m["kind"] = kind or "cagra-hnsw"
print(json.dumps(m))
'''


@pytest.fixture
def stubdir(tmp_path):
    p = tmp_path / "stub.py"
    p.write_text(STUB)
    return tmp_path


def job(attempt: int = 2) -> Job:
    return Job(build_id="4:7", round_id=7, attempt=attempt, n_vectors=1000, dim=128, mem_bytes=1 << 20, dataset="sift-128", n_shards=6)


def argvs(d: Path) -> list[list[str]]:
    return [json.loads(p.read_text()) for p in sorted(d.glob("argv.*"))]


def run(builder, stubdir: Path, mode: str = "ok", kind: str = "", cancel: threading.Event | None = None):
    os.environ["XC_DIR"], os.environ["XC_MODE"], os.environ["XC_KIND"] = str(stubdir), mode, kind
    seen: list[float] = []
    try:
        art = builder.build(job(), seen.append, cancel or threading.Event())
        return art, seen
    finally:
        for k in ("XC_DIR", "XC_MODE", "XC_KIND"):
            os.environ.pop(k, None)


def mk(name: str, stubdir: Path, threads: int = 6):
    return by_name(name, Model(), 1.0, cmd=[sys.executable, str(stubdir / "stub.py")], threads=threads)


# -------------------------------------------------------------------- config


def test_xcheck_config_accepts_cagra_refuses_other():
    c = Config.from_env({"DATABASE_URL": "postgres://unused", "BUILDER": "cagra"})
    print(f"BUILDER=cagra -> Config.builder={c.builder!r} builder_cmd={c.builder_cmd} build_threads={c.build_threads}")
    print("spec: decision 10: BUILDER=cagra on the worker; Phase 4 F3 / ConfigError docstring: a bad BUILDER is refused (exit 2)")
    assert c.builder == "cagra"
    for bad in ("other", "CAGRA", "cuvs", "cagra-hnsw", " cagra"):
        with pytest.raises(ConfigError) as ei:
            Config.from_env({"DATABASE_URL": "postgres://unused", "BUILDER": bad})
        print(f"BUILDER={bad!r} -> ConfigError({ei.value})")


def _closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_xcheck_worker_process_exit_codes_for_builder():
    url = f"postgres://x:y@127.0.0.1:{_closed_port()}/none?connect_timeout=1"
    base = {k: v for k, v in os.environ.items() if k not in ("BUILDER", "DATABASE_URL")}
    p = subprocess.run([sys.executable, "-m", "kgpu_worker"], env={**base, "DATABASE_URL": url, "BUILDER": "other", "LOG_FORMAT": "text"},
                       capture_output=True, text=True, timeout=30, cwd=WORKER_DIR)
    err = [l for l in p.stderr.splitlines() if l.strip()]
    print(f"BUILDER=other -> exit {p.returncode}; stderr lines {err}")
    assert p.returncode == 2
    assert len(err) == 1 and "BUILDER" in err[0]
    # BUILDER=cagra passes validation: the worker goes on to the database (unreachable), it is not refused with 2.
    proc = subprocess.Popen([sys.executable, "-m", "kgpu_worker"], env={**base, "DATABASE_URL": url, "BUILDER": "cagra", "LOG_FORMAT": "text"},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=WORKER_DIR)
    try:
        code = proc.wait(timeout=4)
    except subprocess.TimeoutExpired:
        code = None
    proc.kill()
    out = (proc.stdout.read() if proc.stdout else "")[-400:]
    print(f"BUILDER=cagra -> {'still running after 4 s' if code is None else f'exit {code}'}; output tail {out!r}")
    assert code != 2, "BUILDER=cagra must pass the configuration check"


# -------------------------------------------------------------------- the builder


def test_xcheck_by_name_cagra_is_the_same_child_with_the_flag_last(stubdir):
    b = mk("cagra", stubdir)
    print(f"by_name('cagra') -> {type(b).__name__}, name={getattr(b, 'name', None)!r}")
    print("spec: brief: a builder named 'cagra-hnsw' running the builder command with --builder cagra at the end of the argv; decision 10: the same child-process path as hnswlib")
    assert b.name == "cagra-hnsw"
    art, seen = run(b, stubdir)
    av = argvs(stubdir)
    print(f"child argv: {av}; progress seen {seen}; artifact {art}")
    assert len(av) == 1
    a = av[0]
    assert a[0] == "build"
    assert a[-2:] == ["--builder", "cagra"], a
    assert a.count("--builder") == 1
    for flag, val in (("--build-id", "4:7"), ("--attempt", "2"), ("--dataset", "sift-128"), ("--n-shards", "6"), ("--threads", "6")):
        assert a[a.index(flag) + 1] == val, (flag, a)
    for f in stubdir.glob("argv.*"):
        f.unlink()
    _, seen_h = run(mk("hnswlib", stubdir), stubdir)
    print(f"progress the same stub feeds an hnswlib builder: {seen_h} (invariant 9: nothing else differs)")
    assert seen[:3] == [0.25, 0.5, 1.0], "stderr progress feeds the callback"
    assert seen == seen_h, "the cagra builder's progress handling must be the hnswlib one"
    assert isinstance(art, Artifact) and art.kind == "cagra-hnsw" and art.index_bytes == 4242
    assert art.build_id == "4:7" and art.attempt == 2


def test_xcheck_hnswlib_argv_unchanged(stubdir):
    b = mk("hnswlib", stubdir)
    art, _ = run(b, stubdir, kind="-")
    a = argvs(stubdir)[0]
    print(f"by_name('hnswlib') name={getattr(b, 'name', None)!r}; child argv {a}; artifact kind {art.kind!r} (manifest without 'kind')")
    print("spec: Phase 4 decision 18 unchanged: no --builder cagra for hnswlib; whether it passes --builder hnswlib explicitly is not specified (logged)")
    assert "cagra" not in a
    assert art.kind != "cagra-hnsw"


def test_xcheck_cagra_kind_comes_from_the_manifest(stubdir):
    print("spec: notes 'the Artifact's kind from the manifest'; docstring: 'the Artifact's kind is the manifest's \"cagra-hnsw\"'")
    b = mk("cagra", stubdir)
    art, _ = run(b, stubdir, kind="manifest-says-this")
    print(f"manifest kind 'manifest-says-this' -> Artifact.kind {art.kind!r}")
    assert art.kind == "manifest-says-this"
    for f in stubdir.glob("argv.*"):
        f.unlink()
    art2, _ = run(b, stubdir, kind="-")
    print(f"manifest without 'kind' (a pre-Phase-5 builder) -> Artifact.kind {art2.kind!r} (spec silent on the fallback; logged)")


def test_xcheck_cagra_gate_failure_and_cancel_unchanged(stubdir):
    b = mk("cagra", stubdir)
    with pytest.raises(GateRefused) as ei:
        run(b, stubdir, mode="gate")
    print(f"exit 3 -> GateRefused({ei.value})")
    with pytest.raises(Exception) as e1:
        run(b, stubdir, mode="fail")
    print(f"exit 1 -> {type(e1.value).__name__}({e1.value})")
    assert "Boom: one line" in str(e1.value), "the failure document's reason reaches the worker"
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        run(b, stubdir, mode="hang", cancel=cancel)
    dt = time.monotonic() - t0
    print(f"cancel after 0.5 s -> Cancelled in {dt:.2f}s (the stub would hang 60 s)")
    assert dt < 10


def test_xcheck_cagra_against_the_real_cli_on_a_laptop(tmp_path):
    """The refusal path end to end: the worker's own default builder command
    (the installed kgpu_builder) with BUILDER=cagra on a machine without cuVS."""
    try:
        import cuvs  # noqa: F401
        pytest.skip("cuVS importable here")
    except ImportError:
        pass
    c = Config.from_env({"DATABASE_URL": "postgres://unused", "BUILDER": "cagra", "BUILD_THREADS": "6"})
    b = by_name("cagra", c.model, 1.0, cmd=c.builder_cmd, threads=c.build_threads)
    env_keep = {k: os.environ.get(k) for k in ("S3_ENDPOINT",)}
    os.environ["S3_ENDPOINT"] = f"http://127.0.0.1:{_closed_port()}"
    t0 = time.monotonic()
    try:
        with pytest.raises(Exception) as ei:
            b.build(job(), lambda p: None, threading.Event())
    finally:
        for k, v in env_keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    dt = time.monotonic() - t0
    print(f"builder_cmd {c.builder_cmd} + cagra -> {type(ei.value).__name__}: {str(ei.value)[:300]} in {dt:.2f}s")
    print("spec: decision 10: without cuVS the CLI refuses in one line (exit 1), 'what every laptop and kind run sees'; the worker's fail path carries the reason")
    assert "CuvsUnavailable" in str(ei.value)
    assert not isinstance(ei.value, GateRefused)
    assert dt < 10


# -------------------------------------------------------------------- the GPU image (text only)


def test_xcheck_dockerfile_gpu_text():
    p = WORKER_DIR / "Dockerfile.gpu"
    print("spec: decision 10: worker/Dockerfile.gpu FROM rapidsai/base:26.10-cuda12-py3.12, the builder and the worker installed on top, BUILDER=cagra; decision 1: the CPU image unchanged")
    assert p.exists(), "worker/Dockerfile.gpu missing"
    lines = [l.strip() for l in p.read_text().splitlines()]
    froms = [l for l in lines if l.upper().startswith("FROM ")]
    envs = [l for l in lines if l.upper().startswith("ENV ")]
    installs = [l for l in lines if any(w in l for w in ("pip", "uv ", "uv\t", "install"))]
    copies = [l for l in lines if l.upper().startswith("COPY ")]
    print(f"FROM lines: {froms}")
    print(f"ENV lines: {envs}")
    print(f"COPY lines: {copies}")
    print(f"install lines: {installs}")
    assert froms and froms[0].split()[1] == "rapidsai/base:26.10-cuda12-py3.12"
    text = p.read_text()
    assert "builder" in text and "worker" in text
    assert any("BUILDER=cagra" in e.replace(" =", "=").replace("= ", "=") or ("BUILDER" in e and "cagra" in e) for e in envs), envs
    blob = " ".join(installs + copies)
    assert "builder" in blob and "worker" in blob, "both packages copied and installed"
    cpu = (WORKER_DIR / "Dockerfile").read_text()
    print(f"CPU image worker/Dockerfile mentions cagra: {'cagra' in cpu}; FROM {[l for l in cpu.splitlines() if l.upper().startswith('FROM ')]}")
    assert "cagra" not in cpu, "decision 1: the CPU image is unchanged"
