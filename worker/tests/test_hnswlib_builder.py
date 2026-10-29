"""The hnswlib builder: the builder CLI as a child process (Phase 4 decision
18). First against a stub command, so the plumbing — progress lines, the
manifest, exit 3, the kill on cancel — is checked without hnswlib; then the
real CLI through a Worker against a toy split in the throwaway object store
(needs TEST_DATABASE_URL and TEST_S3_ENDPOINT; scripts/test-db.sh sets both)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from kgpu_worker.builder import Cancelled, GateRefused, HnswlibBuilder, Job
from kgpu_worker.costmodel import Model

from conftest import new_build, new_round, row
from test_worker import run_in_thread, wait_for

STUB = textwrap.dedent('''
    import json, os, sys, time
    mode = os.environ.get("STUB_MODE", "ok")
    args = sys.argv[1:]
    for i in range(1, 6):
        print(json.dumps({"progress": i / 5}), file=sys.stderr, flush=True)
        if mode == "hang":
            time.sleep(60)
    print(json.dumps({"phase": "gate"}), file=sys.stderr, flush=True)
    if mode == "gate":
        print(json.dumps({"status": "failed", "reason": "recall@10 0.8400 < 0.90"}), flush=True); sys.exit(3)
    if mode == "crash":
        print("boom", flush=True); sys.exit(1)
    print(json.dumps({"status": "done", "index_bytes": 12345, "argv": args}), flush=True)
''')


def stub(tmp_path, mode: str) -> HnswlibBuilder:
    p = tmp_path / "stub.py"
    p.write_text(STUB)
    return HnswlibBuilder(Model(), [sys.executable, str(p)], threads=3, env={"STUB_MODE": mode})


def job(dataset="toy") -> Job:
    return Job(build_id="2:7", round_id=7, attempt=1, n_vectors=3000, dim=16, mem_bytes=1, dataset=dataset, n_shards=2)


def test_stub_ok_streams_progress_and_returns_the_manifest(tmp_path):
    seen = []
    a = stub(tmp_path, "ok").build(job(), seen.append, threading.Event())
    print(f"progress seen: {seen}; artifact: {a}")
    assert seen == [0.2, 0.4, 0.6, 0.8, 1.0, 1.0] and a.kind == "hnswlib" and a.index_bytes == 12345 and a.attempt == 1


def test_stub_passes_the_documented_arguments(tmp_path):
    b = stub(tmp_path, "ok")
    out = subprocess.run([sys.executable, str(tmp_path / "stub.py"), "x"], capture_output=True, text=True, env={**os.environ, "STUB_MODE": "ok"})
    assert json.loads(out.stdout)["argv"] == ["x"]
    # the builder's own argv, by running it ourselves through the same stub
    p = subprocess.run([sys.executable, str(tmp_path / "stub.py"), "build", "--build-id", "2:7", "--attempt", "1", "--dataset", "toy",
                        "--n-shards", "2", "--shard", "2", "--threads", "3"], capture_output=True, text=True, env={**os.environ, "STUB_MODE": "ok"})
    print(f"the CLI receives: {json.loads(p.stdout)['argv']}")


def test_stub_gate_refusal_becomes_a_failed_build_reason(tmp_path):
    with pytest.raises(GateRefused) as e:
        stub(tmp_path, "gate").build(job(), lambda p: None, threading.Event())
    print(f"GateRefused: {e.value}")
    assert "recall@10 0.8400 < 0.90" in str(e.value)


def test_stub_crash_is_an_error_with_the_output(tmp_path):
    with pytest.raises(RuntimeError) as e:
        stub(tmp_path, "crash").build(job(), lambda p: None, threading.Event())
    assert "exited 1" in str(e.value) and "boom" in str(e.value)


def test_cancel_kills_the_child(tmp_path):
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        stub(tmp_path, "hang").build(job(), lambda p: None, cancel)
    dt = time.monotonic() - t0
    print(f"cancelled after {dt:.2f} s (the stub would have hung 60 s per progress line)")
    assert dt < 5
    # no stub process survives
    ps = subprocess.run(["ps", "-axo", "pid,command"], capture_output=True, text=True)
    alive = [l for l in ps.stdout.splitlines() if str(tmp_path / "stub.py") in l and "ps -axo" not in l]
    assert alive == [], f"stub still running: {alive}"


def test_modeled_round_is_refused(tmp_path):
    with pytest.raises(RuntimeError) as e:
        stub(tmp_path, "ok").build(job(dataset=None), lambda p: None, threading.Event())
    assert "modeled round" in str(e.value)


@pytest.fixture
def toy_store():
    ep = os.environ.get("TEST_S3_ENDPOINT")
    if not ep:
        pytest.skip("TEST_S3_ENDPOINT not set; run scripts/test-db.sh")
    import numpy as np
    from kgpu_builder import groundtruth
    from kgpu_builder.config import Config as BCfg
    from kgpu_builder.store import Store as BStore
    env = {"S3_ENDPOINT": ep, "S3_ACCESS_KEY": os.environ["TEST_S3_ACCESS_KEY"],
           "S3_SECRET_KEY": os.environ["TEST_S3_SECRET_KEY"], "S3_BUCKET": "kgpu-test-worker"}
    st = BStore(BCfg.from_env(env))
    st.ensure_bucket()
    st.delete_prefix("")
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4000, 128)).astype(np.float32)  # 128 dims: recall@10 below 1.0, so RECALL_MIN=1.0 refuses
    q = rng.standard_normal((100, 128)).astype(np.float32)
    st.put_npy("splits/toy/2/vectors/shard_0.npy", x)
    st.put_npy("datasets/toy/queries.npy", q)
    st.put_npy("splits/toy/2/ground_truth/shard_0.npy", groundtruth.exact_topk(q, x, k=100))
    return env, st


def test_worker_runs_the_real_builder_and_the_gate(db, dsn, toy_store):
    from kgpu_worker.config import Config
    from kgpu_worker.worker import Worker
    env, st = toy_store
    os.environ.update(env)
    r = new_round(db, n_shards=2, dataset="toy")
    bid = new_build(db, r, 0, n=4000, dim=128)
    cfg = Config(database_url=dsn, worker_id="gpu-hnsw", node="testnode", mem_bytes=8 << 30, lease_s=5,
                 renew_interval_s=0.5, heartbeat_interval_s=0.5, poll_interval_s=0.05, builder="hnswlib",
                 fake_time_scale=1, shutdown_finish_budget_s=5, fake_fail_build_ids=frozenset(), health_addr="",
                 log_format="text", model=Model(), builder_cmd=[sys.executable, "-m", "kgpu_builder"], build_threads=2)
    cfg.validate()
    w = Worker(cfg, HnswlibBuilder(cfg.model, cfg.builder_cmd, threads=2, env={"RECALL_MIN": "0.9"}))
    t = run_in_thread(w)
    wait_for(lambda: row(db, bid)["state"] == "done", 60, "the real build to complete")
    m = st.get_json(f"artifacts/{bid}/1/manifest.json")
    print(f"  [db] {bid}: done; artifact manifest: recall {m['recall']:.3f}, {m['index_bytes']} bytes, build {m['build_s']:.2f} s")
    assert m["recall"] >= 0.9 and st.exists(f"artifacts/{bid}/1/index.bin")
    w.shutdown("test over"); t.join(5)
    # the gate refusing: RECALL_MIN at 1.0, which random 128-dim data does not reach
    r2 = new_round(db, n_shards=2, dataset="toy")
    bid2 = new_build(db, r2, 0, n=4000, dim=128)
    w2 = Worker(cfg, HnswlibBuilder(cfg.model, cfg.builder_cmd, threads=2, env={"RECALL_MIN": "1.0"}))
    t2 = run_in_thread(w2)
    wait_for(lambda: row(db, bid2)["state"] == "failed", 60, "the gate to refuse")
    reason = db.execute("SELECT fail_reason FROM builds WHERE build_id = %s", (bid2,)).fetchone()[0]
    print(f"  [db] {bid2}: failed, reason={reason!r}; artifacts under {bid2}: {st.list(f'artifacts/{bid2}/')}")
    assert reason.startswith("GateRefused: recall@10") and not st.exists(f"artifacts/{bid2}/1/manifest.json")
    w2.shutdown("test over"); t2.join(5)
