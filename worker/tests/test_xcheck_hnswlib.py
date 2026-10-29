"""Cross-check (black box) of Phase 4 step 2, the worker side (decision 18).

Spec only: Phase 4 decisions 15, 17, 18; the docstrings of
kgpu_worker.builder (HnswlibBuilder: "stderr progress lines feed progress;
cancel kills the child; exit 3 becomes GateRefused; exit 0's manifest
becomes the Artifact. The object store's address and keys reach the child
through the environment (S3_*), as does RECALL_MIN.") and kgpu_worker.config
(BUILDER, BUILDER_CMD, BUILD_THREADS, RECALL_MIN, S3_*; "Every value is
validated before the worker connects to anything; a bad value exits with
status 2 and one line saying which variable. A worker must never claim a
build it cannot handle.").

The builder is replaced by stub scripts that honour decision 17's contract
and record their argv, environment and pid; the worker runs as a process
(`python -m kgpu_worker`) against the throwaway Postgres.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from kgpu_worker.builder import Cancelled, HnswlibBuilder, Job
from kgpu_worker.config import Config

from conftest import new_build, new_round, row

WORKER_DIR = Path(__file__).resolve().parents[1]

STUB_HEAD = """#!/bin/bash
DIR={dir!r}
printf '%s\\n' "$@" > "$DIR/argv.$$"
echo "$$" >> "$DIR/pids"
env > "$DIR/env.$$"
BID=; ATT=
while [ $# -gt 0 ]; do case "$1" in --build-id) BID=$2; shift 2;; --attempt) ATT=$2; shift 2;; *) shift;; esac; done
manifest() {{ printf '{{"status":"done","kind":"hnswlib","build_id":"%s","attempt":%s,"n_vectors":1000,"dim":128,"index_bytes":4096,"index_sha256":"00","recall":0.99,"build_s":0.1}}\\n' "$BID" "$ATT"; }}
"""


def stub(dir_: Path, body: str, name: str = "stub.sh") -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    p = dir_ / name
    p.write_text(STUB_HEAD.format(dir=str(dir_)) + body)
    p.chmod(0o755)
    return p


def runs(dir_: Path) -> list[tuple[list[str], dict[str, str]]]:
    out = []
    for a in sorted(dir_.glob("argv.*")):
        pid = a.name.split(".", 1)[1]
        env = {}
        for line in (dir_ / f"env.{pid}").read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                env[k] = v
        out.append((a.read_text().splitlines(), env))
    return out


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(st) and not st.startswith("Z")


def model():
    return Config.from_env({"DATABASE_URL": "postgres://unused"}).model


# ---------------------------------------------------------------- unit level


def test_xcheck_hnswlib_env_passthrough_and_argv(tmp_path, monkeypatch):
    d = tmp_path / "s"
    cmd = stub(d, 'echo \'{"progress": 0.5}\' >&2\necho \'{"phase": "gate"}\' >&2\nmanifest\nexit 0\n')
    monkeypatch.setenv("RECALL_MIN", "0.77")
    monkeypatch.setenv("S3_ENDPOINT", "http://objects.example:9000")
    monkeypatch.setenv("S3_ACCESS_KEY", "ak")
    monkeypatch.setenv("S3_SECRET_KEY", "sk")
    monkeypatch.setenv("S3_BUCKET", "bk")
    b = HnswlibBuilder(model(), [str(cmd)], threads=3)
    seen: list[float] = []
    art = b.build(Job("7:42", 42, 2, 1000, 128, 1 << 20, dataset="sift-128", n_shards=50), seen.append, threading.Event())
    (argv, env), = runs(d)
    print(f"artifact {art}; progress callbacks {seen}")
    print(f"child argv: {argv}")
    print(f"child env: RECALL_MIN={env.get('RECALL_MIN')} S3_ENDPOINT={env.get('S3_ENDPOINT')} S3_BUCKET={env.get('S3_BUCKET')} S3_ACCESS_KEY={env.get('S3_ACCESS_KEY')}")
    assert argv == ["build", "--build-id", "7:42", "--attempt", "2", "--dataset", "sift-128", "--n-shards", "50", "--shard", "7", "--threads", "3"], "decision 17 argv"
    assert env.get("RECALL_MIN") == "0.77" and env.get("S3_ENDPOINT") == "http://objects.example:9000" and env.get("S3_BUCKET") == "bk"
    assert art.build_id == "7:42" and art.attempt == 2 and art.index_bytes == 4096
    assert 0.5 in seen


def test_xcheck_hnswlib_explicit_env_argument(tmp_path, monkeypatch):
    """HnswlibBuilder takes env: is it the child's whole environment or an
    overlay on the worker's? The docstring does not say. Recorded."""
    d = tmp_path / "s"
    cmd = stub(d, "manifest\nexit 0\n")
    monkeypatch.setenv("RECALL_MIN", "0.77")
    monkeypatch.setenv("XCHECK_MARK", "inherited")
    b = HnswlibBuilder(model(), [str(cmd)], threads=1, env={"RECALL_MIN": "0.5", "S3_ENDPOINT": "http://given:1"})
    b.build(Job("1:1", 1, 1, 1000, 128, 1, dataset="sift-128", n_shards=6), lambda p: None, threading.Event())
    (_, env), = runs(d)
    print(f"env={{RECALL_MIN: 0.5, S3_ENDPOINT: given}} with RECALL_MIN=0.77 and XCHECK_MARK in the worker's environment -> child sees RECALL_MIN={env.get('RECALL_MIN')}, S3_ENDPOINT={env.get('S3_ENDPOINT')}, XCHECK_MARK={env.get('XCHECK_MARK')}, PATH set: {'PATH' in env}")
    assert env.get("RECALL_MIN") == "0.5" and env.get("S3_ENDPOINT") == "http://given:1"


def test_xcheck_hnswlib_builder_cmd_quoting(tmp_path):
    d = tmp_path / "dir with space"
    cmd = stub(d, "manifest\nexit 0\n", name="my stub.sh")
    raw = f"'{cmd}' --flag 'two words' \"x y\""
    cfg = Config.from_env({"DATABASE_URL": "postgres://unused", "BUILDER": "hnswlib", "BUILDER_CMD": raw})
    print(f"BUILDER_CMD={raw!r} -> builder_cmd {cfg.builder_cmd}")
    b = HnswlibBuilder(cfg.model, cfg.builder_cmd, threads=1)
    b.build(Job("0:5", 5, 1, 1000, 128, 1, dataset="sift-128", n_shards=6), lambda p: None, threading.Event())
    (argv, _), = runs(d)
    print(f"child argv: {argv}")
    assert cfg.builder_cmd == [str(cmd), "--flag", "two words", "x y"]
    assert argv[:3] == ["--flag", "two words", "x y"] and argv[3] == "build"


def test_xcheck_hnswlib_cancel_kills_a_term_ignoring_child_and_its_grandchild(tmp_path):
    """Decision 18: cancel kills the child (SIGKILL), 'parity with the shard:
    the same process, the same kill' (the shard kills the process group,
    decision 19). A BUILDER_CMD such as `uv run python -m kgpu_builder` makes
    the real builder a grandchild, so the grandchild matters."""
    d = tmp_path / "s"
    cmd = stub(d, """trap '' TERM INT HUP
sleep 1000 &
echo $! > "$DIR/grandchild"
while true; do echo '{"progress": 0.1}' >&2; sleep 0.2; done
""")
    b = HnswlibBuilder(model(), [str(cmd)], threads=1)
    cancel = threading.Event()
    first = threading.Event()
    result: dict = {}

    def run():
        try:
            result["art"] = b.build(Job("0:6", 6, 1, 1000, 128, 1, dataset="sift-128", n_shards=6), lambda p: first.set(), cancel)
        except BaseException as e:  # noqa: BLE001
            result["exc"] = e
        result["t"] = time.monotonic()

    th = threading.Thread(target=run, daemon=True)
    th.start()
    assert first.wait(10), "no progress callback from the stub"
    child = int((d / "pids").read_text().split()[0])
    grand = int((d / "grandchild").read_text())
    t0 = time.monotonic()
    cancel.set()
    th.join(15)
    returned = result.get("t", float("inf")) - t0
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and (alive(child) or alive(grand)):
        time.sleep(0.05)
    c_alive, g_alive = alive(child), alive(grand)
    print(f"cancel: build() returned after {returned:.2f}s with {type(result.get('exc')).__name__}; child {child} alive={c_alive}; grandchild {grand} alive={g_alive}")
    if g_alive:
        os.kill(grand, signal.SIGKILL)
    if c_alive:
        os.kill(child, signal.SIGKILL)
    assert isinstance(result.get("exc"), Cancelled)
    assert returned < 5
    assert not c_alive, "decision 18: cancel kills the child"
    assert not g_alive, "decision 18 'parity with the shard' (which kills the group): the grandchild survived the cancel"


@pytest.mark.parametrize("case", ["exit1_traceback_empty_stdout", "exit0_unparsable_stdout", "exit3_without_document"])
def test_xcheck_hnswlib_odd_child_outputs(tmp_path, case):
    """Decision 17's contract broken by the child in ways the real CLI has
    been seen to (exit 1 with a traceback and empty stdout: --recall-k 200,
    RECALL_MIN=abc) or might. The worker must raise something it can mark
    failed with a readable reason, never return an Artifact."""
    d = tmp_path / "s"
    body = {
        "exit1_traceback_empty_stdout": "echo 'Traceback (most recent call last):' >&2\necho 'ValueError: k=200 exceeds the stored width' >&2\nexit 1\n",
        "exit0_unparsable_stdout": "echo 'not a manifest'\nexit 0\n",
        "exit3_without_document": "exit 3\n",
    }[case]
    cmd = stub(d, body)
    b = HnswlibBuilder(model(), [str(cmd)], threads=1)
    try:
        art = b.build(Job("0:8", 8, 1, 1000, 128, 1, dataset="sift-128", n_shards=6), lambda p: None, threading.Event())
        print(f"{case}: returned an Artifact {art}")
        pytest.fail(f"{case}: a broken child produced an Artifact")
    except Cancelled:
        pytest.fail("not a cancellation")
    except Exception as e:  # noqa: BLE001
        print(f"{case}: {type(e).__name__}: {str(e)[:300]!r}")
        assert str(e).strip(), "the reason should be readable, not empty"


# ---------------------------------------------------------------- the worker process


class WorkerProc:
    def __init__(self, dsn: str, env_extra: dict[str, str], wid: str = "xc-hnsw"):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("S3_", "RECALL_MIN", "BUILDER", "BUILD_THREADS"))}
        env.update(DATABASE_URL=dsn, WORKER_ID=wid, HEALTH_ADDR="", LOG_FORMAT="text", POLL_INTERVAL_SECONDS="0.1",
                   HEARTBEAT_INTERVAL_SECONDS="0.5", LEASE_SECONDS="30", RENEW_INTERVAL_SECONDS="5")
        env.update(env_extra)
        self.p = subprocess.Popen([sys.executable, "-m", "kgpu_worker"], cwd=WORKER_DIR, env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.out: list[str] = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        assert self.p.stdout
        for line in self.p.stdout:
            self.out.append(line.rstrip())

    def stop(self):
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
            try:
                self.p.wait(15)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()


def wait_state(conn, build_id: str, states: set[str], timeout: float = 30) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = conn.execute("SELECT state FROM builds WHERE build_id=%s", (build_id,)).fetchone()[0]
        if st in states:
            break
        time.sleep(0.1)
    r = row(conn, build_id)
    r["fail_reason"] = conn.execute("SELECT fail_reason FROM builds WHERE build_id=%s", (build_id,)).fetchone()[0]
    print(f"  [db] fail_reason={r['fail_reason']!r}")
    return r


@pytest.mark.parametrize("var,val,must", [
    ("RECALL_MIN", "abc", True), ("BUILD_THREADS", "-3", True), ("BUILDER", "nope", True),
    ("BUILDER_CMD", "'unterminated", True), ("RECALL_MIN", "1.5", False), ("RECALL_MIN", "-1", False),
    ("BUILD_THREADS", "0", False),
])
def test_xcheck_worker_refuses_bad_builder_config(var, val, must):
    """config docstring: 'a bad value exits with status 2 and one line saying
    which variable', before connecting (DATABASE_URL points nowhere)."""
    w = WorkerProc("postgres://xc@127.0.0.1:1/none?connect_timeout=1", {"BUILDER": "hnswlib", var: val})
    try:
        rc = w.p.wait(10)
    except subprocess.TimeoutExpired:
        rc = None
    w.stop()
    time.sleep(0.2)
    print(f"{var}={val!r}: exit {rc}; output ({len(w.out)} lines): {w.out[:3]}")
    if must:
        assert rc == 2, f"{var}={val!r} is not a usable value: the worker must refuse it with exit 2"
        assert len(w.out) == 1 and var in w.out[0], "one line naming the variable"
    else:
        print(f"  (spec silent on {var}={val!r}; recorded only)")


def test_xcheck_worker_process_argv_env_and_completion(db, dsn, tmp_path):
    d = tmp_path / "s"
    cmd = stub(d, 'for p in 0.2 0.6 1.0; do echo "{\\"progress\\": $p}" >&2; sleep 0.2; done\necho \'{"phase": "gate"}\' >&2\nmanifest\nexit 0\n')
    r = new_round(db, n_shards=50, dataset="sift-128")
    b = new_build(db, r, 7)
    w = WorkerProc(dsn, {"BUILDER": "hnswlib", "BUILDER_CMD": str(cmd), "BUILD_THREADS": "3", "RECALL_MIN": "0.77",
                         "S3_ENDPOINT": "http://objects.example:9000", "S3_BUCKET": "bk", "S3_ACCESS_KEY": "ak", "S3_SECRET_KEY": "sk"})
    try:
        got = wait_state(db, b, {"done", "failed"})
    finally:
        w.stop()
    (argv, env), = runs(d)
    print(f"child argv: {argv}\nchild env: RECALL_MIN={env.get('RECALL_MIN')} S3_ENDPOINT={env.get('S3_ENDPOINT')} S3_BUCKET={env.get('S3_BUCKET')}")
    assert got["state"] == "done" and got["attempt"] == 1
    assert argv == ["build", "--build-id", b, "--attempt", "1", "--dataset", "sift-128", "--n-shards", "50", "--shard", "7", "--threads", "3"]
    assert env.get("RECALL_MIN") == "0.77" and env.get("S3_ENDPOINT") == "http://objects.example:9000" and env.get("S3_BUCKET") == "bk"


def test_xcheck_worker_process_modeled_round_under_hnswlib(db, dsn, tmp_path):
    """BUILDER=hnswlib needs a dataset (config docstring: 'the round must
    carry a dataset'). A modeled round's build must end failed with a
    readable reason, and the builder must not run."""
    d = tmp_path / "s"
    cmd = stub(d, "manifest\nexit 0\n")
    r = new_round(db, n_shards=6, dataset=None)
    b = new_build(db, r, 2)
    w = WorkerProc(dsn, {"BUILDER": "hnswlib", "BUILDER_CMD": str(cmd)})
    try:
        got = wait_state(db, b, {"done", "failed"}, timeout=20)
    finally:
        w.stop()
    print(f"worker log tail: {w.out[-5:]}")
    print(f"stub runs: {len(runs(d))}")
    assert got["state"] == "failed", "a modeled round under the hnswlib builder cannot be built"
    assert got["fail_reason"] and "dataset" in got["fail_reason"].lower()
    assert runs(d) == []


def test_xcheck_worker_process_real_cli_missing_split_fails_readably(db, dsn):
    """The real CLI (default BUILDER_CMD) against an empty bucket: exit 1 with
    'object not in the store' (decision 17's 1 = anything else); the worker
    marks the build failed with that reason."""
    s3 = os.environ.get("TEST_S3_ENDPOINT")
    if not s3:
        pytest.skip("TEST_S3_ENDPOINT unset")
    import uuid
    from kgpu_builder.config import Config as BConfig
    from kgpu_builder.store import Store as BStore
    bucket = "xcw-" + uuid.uuid4().hex[:8]
    ak, sk = os.environ.get("TEST_S3_ACCESS_KEY", ""), os.environ.get("TEST_S3_SECRET_KEY", "")
    BStore(BConfig(data_dir=Path("/tmp"), s3_endpoint=s3, s3_access_key=ak, s3_secret_key=sk, s3_bucket=bucket, s3_region="us-east-1")).ensure_bucket()
    r = new_round(db, n_shards=6, dataset="sift-128")
    b = new_build(db, r, 3)
    w = WorkerProc(dsn, {"BUILDER": "hnswlib", "S3_ENDPOINT": s3, "S3_ACCESS_KEY": ak, "S3_SECRET_KEY": sk, "S3_BUCKET": bucket})
    try:
        got = wait_state(db, b, {"done", "failed"}, timeout=60)
    finally:
        w.stop()
    print(f"worker log tail: {w.out[-4:]}")
    assert got["state"] == "failed"
    assert "splits/sift-128/6/vectors/shard_3.npy" in (got["fail_reason"] or "")
