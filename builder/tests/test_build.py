"""The real build on a tiny synthetic split in the throwaway object store:
builds, gates, publishes manifest-last; refuses below the threshold with
nothing written; the CLI's exit codes and streams. Needs TEST_S3_ENDPOINT."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from kgpu_builder import groundtruth
from kgpu_builder.build import BuildSpec, GateFailed, artifact_prefix, run_build
from kgpu_builder.config import Config
from kgpu_builder.store import NotFound, Store


def _env():
    ep = os.environ.get("TEST_S3_ENDPOINT")
    if not ep:
        pytest.skip("TEST_S3_ENDPOINT not set; run scripts/test-db.sh")
    return {"S3_ENDPOINT": ep, "S3_ACCESS_KEY": os.environ["TEST_S3_ACCESS_KEY"],
            "S3_SECRET_KEY": os.environ["TEST_S3_SECRET_KEY"], "S3_BUCKET": "kgpu-test-build"}


@pytest.fixture
def toy():
    """A 'toy' dataset split 2 ways: shard 0 holds 3000 x 16 vectors, with the
    exact top-100 of 200 queries, laid out as decision 9 says."""
    env = _env()
    st = Store(Config.from_env(env))
    st.ensure_bucket()
    st.delete_prefix("")
    rng = np.random.default_rng(7)
    x = rng.standard_normal((4000, 128)).astype(np.float32)  # 128 dims: recall@10 lands below 1.0, so RECALL_MIN=1.0 refuses
    q = rng.standard_normal((200, 128)).astype(np.float32)
    st.put_npy("splits/toy/2/vectors/shard_0.npy", x)
    st.put_npy("datasets/toy/queries.npy", q)
    st.put_npy("splits/toy/2/ground_truth/shard_0.npy", groundtruth.exact_topk(q, x, k=100))
    return env, st, x, q


def test_build_gates_and_publishes_manifest_last(toy):
    env, st, x, q = toy
    seen = []
    orig_bytes, orig_file = st.put_bytes, st.put_file
    st.put_bytes = lambda name, data, ct="application/octet-stream": (seen.append(name), orig_bytes(name, data, ct))[1]  # record the write order
    st.put_file = lambda name, path: (seen.append(name), orig_file(name, path))[1]
    progress, phases = [], []
    m = run_build(st, BuildSpec("0:1", 1, "toy", 2, 0, threads=2), progress.append, phases.append)
    print(f"recall@10 {m['recall']:.3f}, index {m['index_bytes']} bytes, build {m['build_s']:.2f} s; writes in order: {seen}; phases {phases}")
    assert m["status"] == "done" and m["kind"] == "hnswlib" and m["recall"] >= 0.9
    assert m["n_vectors"] == 4000 and m["dim"] == 128 and m["attempt"] == 1 and m["recall"] < 1.0
    assert seen == [f"{artifact_prefix('0:1', 1)}/index.bin", f"{artifact_prefix('0:1', 1)}/manifest.json"], "index first, manifest last (invariant 7)"
    assert progress[-1] == 1.0 and len(progress) == 10 and progress == sorted(progress)
    assert phases == ["build", "gate", "publish"]
    assert st.get_json("artifacts/0:1/1/manifest.json") == m
    blob = st.get_bytes("artifacts/0:1/1/index.bin")
    assert len(blob) == m["index_bytes"]
    import hashlib, hnswlib, tempfile
    assert hashlib.sha256(blob).hexdigest() == m["index_sha256"]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "i.bin"); open(p, "wb").write(blob)
        idx = hnswlib.Index(space="l2", dim=128); idx.load_index(p, max_elements=4000)
        idx.set_ef(100)
        labels, _ = idx.knn_query(q, k=10)
        truth = st.get_npy("splits/toy/2/ground_truth/shard_0.npy")
        r = groundtruth.recall_at(labels.astype(np.int32), truth, 10)
    print(f"the published index reloads and scores recall@10 {r:.3f} again")
    assert abs(r - m["recall"]) < 1e-9


def test_gate_failure_writes_nothing(toy):
    env, st, x, q = toy
    with pytest.raises(GateFailed) as e:
        run_build(st, BuildSpec("0:2", 1, "toy", 2, 0, recall_min=1.01))
    print(f"refused: {e.value}")
    assert st.list("artifacts/") == [], "nothing is written before the gate passes"


def test_cli_exit_codes_and_streams(toy):
    env, st, x, q = toy
    base = [sys.executable, "-m", "kgpu_builder", "build", "--dataset", "toy", "--n-shards", "2", "--shard", "0", "--threads", "1"]
    r = subprocess.run(base + ["--build-id", "0:3", "--attempt", "2"], env={**os.environ, **env}, capture_output=True, text=True)
    print(f"ok build: rc={r.returncode}; stderr lines={len(r.stderr.splitlines())}; stdout={r.stdout[:120]}...")
    assert r.returncode == 0
    m = json.loads(r.stdout)
    assert m["status"] == "done" and m["build_id"] == "0:3" and m["attempt"] == 2
    lines = [json.loads(l) for l in r.stderr.splitlines() if l.startswith("{")]
    assert [l for l in lines if "progress" in l][-1]["progress"] == 1.0
    assert [l["phase"] for l in lines if "phase" in l] == ["build", "gate", "publish"]
    assert st.exists("artifacts/0:3/2/manifest.json")
    r = subprocess.run(base + ["--build-id", "0:4", "--attempt", "1"], env={**os.environ, **env, "RECALL_MIN": "1.0"}, capture_output=True, text=True)
    print(f"gate failure: rc={r.returncode}; stdout={r.stdout.strip()}")
    assert r.returncode == 3
    f = json.loads(r.stdout)
    assert f["status"] == "failed" and "recall@10" in f["reason"] and not st.exists("artifacts/0:4/1/manifest.json")
    r = subprocess.run(base + ["--build-id", "9:9", "--attempt", "1", "--shard", "5"], env={**os.environ, **env}, capture_output=True, text=True)
    print(f"missing shard: rc={r.returncode}; stdout={r.stdout.strip()[:100]}")
    assert r.returncode == 1 and json.loads(r.stdout)["status"] == "failed"
    r = subprocess.run(base + ["--build-id", "0:5", "--attempt", "1"], env={**os.environ, **env, "RECALL_MIN": "1.01"}, capture_output=True, text=True)
    print(f"RECALL_MIN out of range: rc={r.returncode}; stderr={r.stderr.strip()[:100]}")
    assert r.returncode == 2 and r.stdout == "" and "RECALL_MIN" in r.stderr
