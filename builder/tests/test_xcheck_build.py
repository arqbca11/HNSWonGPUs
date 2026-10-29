"""Cross-check (black box) of Phase 4 step 2: the build CLI.

Spec only: Phase 4 decisions 2, 3, 4, 17 (the CLI contract), invariant 7
(index.bin first, manifest.json last, completion after), the flow "The
build, as the worker and the shard both run it (step 2)", `--help` and the
`kgpu_builder.build` docstring ("Nothing is written to the store before the
gate passes ... A kill at any point leaves at most an orphan index.bin").

The CLI is driven only as a process (`python -m kgpu_builder build ...`)
against a throwaway RustFS (TEST_S3_ENDPOINT); the toy splits are written
with the public Store and exact_topk. Every test uses a fresh bucket.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from kgpu_builder.config import Config
from kgpu_builder.groundtruth import exact_topk
from kgpu_builder.store import Store

S3 = os.environ.get("TEST_S3_ENDPOINT")
pytestmark = pytest.mark.skipif(not S3, reason="TEST_S3_ENDPOINT unset")

DOCUMENTED_FIELDS = [  # decision 17: what the manifest records
    "build_id", "attempt", "dataset", "n_shards", "shard", "n_vectors", "dim",
    "m", "ef_construction", "threads", "index_bytes", "index_sha256",
    "recall_k", "ef_search", "build_s", "gate_s", "hnswlib_version",
]


class Toy:
    """A fresh bucket holding one toy dataset split n ways."""

    def __init__(self, tmp: Path, n_shards: int = 2, size: int = 3000, dim: int = 16,
                 n_queries: int = 200, seed: int = 1, dataset: str = "xtoy"):
        self.bucket = "xcb-" + uuid.uuid4().hex[:10]
        self.cfg = Config(
            data_dir=tmp, s3_endpoint=S3,
            s3_access_key=os.environ.get("TEST_S3_ACCESS_KEY", ""),
            s3_secret_key=os.environ.get("TEST_S3_SECRET_KEY", ""),
            s3_bucket=self.bucket, s3_region="us-east-1",
        )
        self.st = Store(self.cfg)
        self.st.ensure_bucket()
        self.dataset, self.n = dataset, n_shards
        rng = np.random.default_rng(seed)
        q = rng.standard_normal((n_queries, dim)).astype(np.float32)
        self.st.put_npy(f"datasets/{dataset}/queries.npy", q)
        for k in range(n_shards):
            v = rng.standard_normal((size, dim)).astype(np.float32)
            self.st.put_npy(f"splits/{dataset}/{n_shards}/vectors/shard_{k}.npy", v)
            self.st.put_npy(f"splits/{dataset}/{n_shards}/ground_truth/shard_{k}.npy", exact_topk(q, v, 100))
        print(f"(bucket {self.bucket}: dataset {dataset}, {n_shards} shards of {size} x {dim}, {n_queries} queries, exact top-100)")

    def env(self, **extra: str) -> dict[str, str]:
        e = {k: v for k, v in os.environ.items() if not k.startswith("S3_") and k != "RECALL_MIN"}
        e.update(S3_ENDPOINT=S3, S3_ACCESS_KEY=self.cfg.s3_access_key, S3_SECRET_KEY=self.cfg.s3_secret_key,
                 S3_BUCKET=self.bucket, S3_REGION="us-east-1")
        e.update(extra)
        return e

    def argv(self, build_id: str = "0:1", attempt: int = 1, shard: int = 0, threads: int = 2,
             dataset: str | None = None, extra: list[str] | None = None) -> list[str]:
        return [sys.executable, "-m", "kgpu_builder", "build", "--build-id", build_id, "--attempt", str(attempt),
                "--dataset", dataset or self.dataset, "--n-shards", str(self.n), "--shard", str(shard),
                "--threads", str(threads)] + (extra or [])

    def run(self, *a, env: dict[str, str] | None = None, **kw) -> subprocess.CompletedProcess:
        argv = self.argv(*a, **kw)
        t0 = time.monotonic()
        p = subprocess.run(argv, capture_output=True, text=True, env=env or self.env(), timeout=600)
        print(f"ran: {' '.join(argv[1:])}  ->  exit {p.returncode} in {time.monotonic() - t0:.2f}s")
        if p.returncode == 2:
            print(f"     usage error on stderr: {p.stderr.strip()[-300:]}")
        return p

    def artifacts(self, prefix: str = "artifacts/") -> list[str]:
        return sorted(self.st.list(prefix))

    def close(self) -> None:
        self.st.delete_prefix("")


@pytest.fixture
def toy(tmp_path):
    t = Toy(tmp_path)
    yield t
    t.close()


def one_json_doc(stdout: str) -> dict:
    """Decision 17: stdout carries ONE JSON document."""
    doc = json.loads(stdout)
    assert isinstance(doc, dict)
    return doc


def json_lines(stderr: str) -> list[dict]:
    return [json.loads(line) for line in stderr.splitlines() if line.strip()]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------------


def test_xcheck_build_success_streams_and_manifest(toy):
    p = toy.run("3:7", 2, shard=1)
    print(f"stdout ({len(p.stdout.splitlines())} line(s)): {p.stdout.strip()[:400]}")
    print(f"stderr lines: {p.stderr.splitlines()}")
    assert p.returncode == 0
    m = one_json_doc(p.stdout)
    lines = json_lines(p.stderr)  # raises if any stderr line is not JSON
    progress = [l["progress"] for l in lines if "progress" in l]
    phases = [l["phase"] for l in lines if "phase" in l]
    print(f"progress values: {progress}; phases: {phases}")
    assert progress == pytest.approx([0.1 * i for i in range(1, 11)]), "decision 17: progress after each of ten add_items batches"
    gate_at = lines.index({"phase": "gate"})
    last_progress_at = max(i for i, l in enumerate(lines) if "progress" in l)
    assert gate_at > last_progress_at, "decision 17: {'phase': 'gate'} comes after the ten progress lines"
    missing = [f for f in DOCUMENTED_FIELDS if f not in m]
    print(f"documented manifest fields missing from stdout: {missing}")
    assert not missing
    assert (m["build_id"], m["attempt"], m["dataset"], m["n_shards"], m["shard"]) == ("3:7", 2, "xtoy", 2, 1)
    assert (m["n_vectors"], m["dim"], m["m"], m["ef_construction"], m["threads"], m["recall_k"], m["ef_search"]) == (3000, 16, 16, 200, 2, 10, 100)
    keys = toy.artifacts()
    print(f"store now holds under artifacts/: {keys}")
    assert keys == ["artifacts/3:7/2/index.bin", "artifacts/3:7/2/manifest.json"]
    stored = toy.st.get_json("artifacts/3:7/2/manifest.json")
    idx = toy.st.get_bytes("artifacts/3:7/2/index.bin")
    print(f"index.bin: {len(idx)} bytes sha256 {sha(idx)[:16]}..; manifest says {m['index_bytes']} / {m['index_sha256'][:16]}..; stored manifest == stdout: {stored == m}")
    assert stored == m, "the stdout document is the artifact manifest (decision 17)"
    assert m["index_bytes"] == len(idx) and m["index_sha256"] == sha(idx)
    assert m["recall"] >= 0.9 if "recall" in m else True


def test_xcheck_build_manifest_names_the_real_hnswlib_version(toy):
    p = toy.run()
    m = one_json_doc(p.stdout)
    installed = importlib.metadata.version("hnswlib")
    print(f"manifest hnswlib_version={m.get('hnswlib_version')!r}; the installed distribution is hnswlib {installed}")
    assert m.get("hnswlib_version") == installed, "decision 17: the manifest records the hnswlib version"


def test_xcheck_build_publish_order_observed_live(tmp_path):
    """Invariant 7 observed from outside: list artifacts/ as fast as possible
    from another thread while a longer build runs."""
    toy = Toy(tmp_path, n_shards=1, size=50_000, dim=32, n_queries=300, seed=5)
    try:
        seen: list[tuple[float, tuple[str, ...]]] = []
        stop = threading.Event()

        def poll() -> None:
            while not stop.is_set():
                ks = tuple(sorted(toy.st.list("artifacts/")))
                if not seen or seen[-1][1] != ks:
                    seen.append((time.monotonic(), ks))

        stderr_at: list[tuple[float, str]] = []
        th = threading.Thread(target=poll, daemon=True)
        th.start()
        proc = subprocess.Popen(toy.argv("0:9", 1), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=toy.env())

        def read_err() -> None:
            assert proc.stderr
            for line in proc.stderr:
                stderr_at.append((time.monotonic(), line.strip()))

        te = threading.Thread(target=read_err, daemon=True)
        te.start()
        out = proc.stdout.read() if proc.stdout else ""
        rc = proc.wait(timeout=600)
        te.join(5)
        time.sleep(0.3)
        stop.set()
        th.join(5)
        t0 = seen[0][0]
        print(f"build exit {rc}; distinct listings the poller saw, in order:")
        for t, ks in seen:
            print(f"  +{t - t0:7.3f}s  {list(ks)}")
        gate_t = next(t for t, l in stderr_at if l == '{"phase": "gate"}')
        print(f"stderr {{'phase': 'gate'}} read at +{gate_t - t0:.3f}s")
        assert rc == 0 and one_json_doc(out)["n_vectors"] == 50_000
        idx_key, man_key = "artifacts/0:9/1/index.bin", "artifacts/0:9/1/manifest.json"
        first_any = next((t for t, ks in seen if ks), None)
        assert first_any is not None and first_any >= gate_t, "docstring: nothing is written before the gate passes"
        for _, ks in seen:
            assert not (man_key in ks and idx_key not in ks), "invariant 7: a manifest was visible without its index"
        intermediate = [ks for _, ks in seen if ks == (idx_key,)]
        print(f"poller observed the index-only state {len(intermediate)} time(s) (the window is the manifest upload)")
        assert seen[-1][1] == (idx_key, man_key)
        if not intermediate:
            pytest.skip("the index-only window was too short to observe; order not contradicted")
    finally:
        toy.close()


def test_xcheck_build_sigkill_mid_build_leaves_no_manifest(tmp_path):
    toy = Toy(tmp_path, n_shards=1, size=50_000, dim=32, n_queries=300, seed=6)
    try:
        proc = subprocess.Popen(toy.argv("0:11", 1), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=toy.env())
        assert proc.stderr
        first = ""
        for line in proc.stderr:
            if "progress" in line:
                first = line.strip()
                break
        os.kill(proc.pid, signal.SIGKILL)
        rc = proc.wait(timeout=30)
        time.sleep(0.5)
        keys = toy.artifacts()
        print(f"killed with SIGKILL after stderr {first!r}; exit {rc}; artifacts/ now holds {keys}")
        assert rc == -signal.SIGKILL
        assert "artifacts/0:11/1/manifest.json" not in keys, "decision 17 flow: a kill leaves no manifest"
        assert keys == [], "docstring: nothing is written before the gate passes (a kill during add_items leaves nothing)"
    finally:
        toy.close()


def test_xcheck_build_second_attempt_never_touches_the_first(toy):
    p1 = toy.run("1:4", 1, shard=1, threads=2)
    assert p1.returncode == 0
    snap1 = {k: sha(toy.st.get_bytes(k)) for k in toy.artifacts()}
    time.sleep(1.1)  # so any rewrite would also change a timestamp
    p2 = toy.run("1:4", 2, shard=1, threads=4)
    assert p2.returncode == 0
    after = {k: sha(toy.st.get_bytes(k)) for k in toy.artifacts()}
    print(f"after attempt 1: {sorted(snap1)}\nafter attempt 2: {sorted(after)}")
    for k, h in snap1.items():
        print(f"  {k}: before {h[:12]}.. after {after.get(k, 'GONE')[:12]}..")
        assert after.get(k) == h, "decision 2: a later attempt never overwrites another attempt's objects"
    assert set(after) - set(snap1) == {"artifacts/1:4/2/index.bin", "artifacts/1:4/2/manifest.json"}
    m2 = toy.st.get_json("artifacts/1:4/2/manifest.json")
    assert m2["attempt"] == 2 and m2["threads"] == 4


def test_xcheck_build_threads_1_vs_4_recall(tmp_path):
    toy = Toy(tmp_path, n_shards=1, size=20_000, dim=32, n_queries=300, seed=7)
    try:
        res = {}
        for th in (1, 4, 1):
            p = toy.run(f"0:{20 + th}", len(res.get(th, [])) + 1, threads=th, extra=["--ef-search", "30"], env=toy.env(RECALL_MIN="0"))
            assert p.returncode == 0, p.stdout
            m = one_json_doc(p.stdout)
            res.setdefault(th, []).append((m["recall"], m["index_sha256"]))
        print(f"threads=1 (twice): {res[1]}\nthreads=4: {res[4]}")
        print(f"threads 1 deterministic across runs: {res[1][0] == res[1][1]}; threads 4 same recall as 1: {res[4][0][0] == res[1][0][0]}; same index bytes: {res[4][0][1] == res[1][0][1]}")
        assert abs(res[4][0][0] - res[1][0][0]) < 0.05, "different thread counts should give comparable recall"
    finally:
        toy.close()


def test_xcheck_build_recall_min_boundary(toy):
    """The gate fails 'below the threshold' (decision 17, roadmap): a recall
    exactly at RECALL_MIN passes; one ulp above refuses."""
    extra = ["--m", "4", "--ef-search", "10"]
    a = one_json_doc(toy.run("0:30", 1, threads=1, extra=extra, env=toy.env(RECALL_MIN="0")).stdout)
    b = one_json_doc(toy.run("0:30", 2, threads=1, extra=extra, env=toy.env(RECALL_MIN="0")).stdout)
    r = a["recall"]
    print(f"measured recall with RECALL_MIN=0, threads 1, twice: {r!r} and {b['recall']!r}")
    assert r == b["recall"], "one thread should reproduce the same index (precondition for an exact boundary)"
    eq = toy.run("0:30", 3, threads=1, extra=extra, env=toy.env(RECALL_MIN=repr(r)))
    print(f"RECALL_MIN={r!r} (exactly the recall): exit {eq.returncode}, stdout {eq.stdout.strip()[:200]}")
    above = math.nextafter(r, 2.0)
    hi = toy.run("0:30", 4, threads=1, extra=extra, env=toy.env(RECALL_MIN=repr(above)))
    print(f"RECALL_MIN={above!r} (one ulp above): exit {hi.returncode}, stdout {hi.stdout.strip()}")
    print(f"artifacts/: {toy.artifacts()}")
    assert eq.returncode == 0
    assert hi.returncode == 3
    doc = one_json_doc(hi.stdout)
    assert doc["status"] == "failed" and doc["reason"].startswith("recall@10 ")
    assert not any(k.startswith("artifacts/0:30/4/") for k in toy.artifacts()), "a refused build publishes nothing"


def test_xcheck_build_gate_reason_is_not_misleading(toy):
    """The gate's reason should read as a true inequality. Decision 17's
    example is 'recall@10 0.84 < 0.90'. Find a recall whose third decimal is
    1-4 (so a threshold just above it rounds DOWN at two decimals), then set
    RECALL_MIN a hair above the recall."""
    r, ef = None, None
    for ef in (5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30):
        m = one_json_doc(toy.run("0:31", ef, threads=1, extra=["--m", "4", "--ef-search", str(ef)], env=toy.env(RECALL_MIN="0")).stdout)
        if round(m["recall"] + 1e-4, 2) < m["recall"]:
            r = m["recall"]
            break
    if r is None:
        pytest.skip("no recall with a rounding-down threshold found")
    thr = r + 1e-4
    p = toy.run("0:31", 100, threads=1, extra=["--m", "4", "--ef-search", str(ef)], env=toy.env(RECALL_MIN=repr(thr)))
    doc = one_json_doc(p.stdout)
    print(f"recall {r!r} at ef_search {ef}, RECALL_MIN {thr!r}: exit {p.returncode}, reason {doc.get('reason')!r}")
    assert p.returncode == 3
    nums = [float(x) for x in doc["reason"].split(" ", 1)[1].split("<")]
    print(f"the reason's two numbers: {nums}; reads as a true inequality: {nums[0] < nums[1]}")
    assert nums[0] < nums[1], "the printed reason should not say e.g. '0.4810 < 0.48' (threshold rounded to two decimals)"


def test_xcheck_build_gate_flags_k_and_ef_search(toy):
    base = one_json_doc(toy.run("0:40", 1, threads=1, extra=["--m", "4"], env=toy.env(RECALL_MIN="0")).stdout)
    k5 = one_json_doc(toy.run("0:40", 2, threads=1, extra=["--m", "4", "--recall-k", "5"], env=toy.env(RECALL_MIN="0")).stdout)
    lo = one_json_doc(toy.run("0:40", 3, threads=1, extra=["--m", "4", "--ef-search", "5"], env=toy.env(RECALL_MIN="0")).stdout)
    for name, m in (("default", base), ("--recall-k 5", k5), ("--ef-search 5", lo)):
        print(f"{name:14s}: recall_k={m['recall_k']} ef_search={m['ef_search']} m={m['m']} recall={m['recall']}")
    assert (base["recall_k"], base["ef_search"]) == (10, 100)
    assert (k5["recall_k"], k5["ef_search"]) == (5, 100)
    assert (lo["recall_k"], lo["ef_search"]) == (10, 5)
    assert lo["recall"] < base["recall"], "a smaller ef_search should search less well"
    # the failure reason names the k actually used
    p = toy.run("0:40", 4, threads=1, extra=["--m", "4", "--recall-k", "5", "--ef-search", "5"], env=toy.env(RECALL_MIN="1.0"))
    d = one_json_doc(p.stdout)
    print(f"--recall-k 5 refused: {d}")
    assert p.returncode == 3 and d["reason"].startswith("recall@5 ")


@pytest.mark.parametrize("case", ["missing_shard", "missing_dataset", "missing_ground_truth", "missing_queries"])
def test_xcheck_build_missing_inputs_exit_1_with_a_failure_document(toy, case):
    shard, dataset = 0, None
    if case == "missing_shard":
        shard = 7
    elif case == "missing_dataset":
        dataset = "no-such-dataset"
    elif case == "missing_ground_truth":
        toy.st.delete(f"splits/xtoy/2/ground_truth/shard_1.npy")
        shard = 1
    else:
        toy.st.delete("datasets/xtoy/queries.npy")
    p = toy.run("5:50", 1, shard=shard, dataset=dataset)
    print(f"{case}: stdout {p.stdout.strip()!r}; stderr tail {p.stderr.strip()[-200:]!r}; artifacts {toy.artifacts()}")
    assert p.returncode == 1, "decision 17: 1 anything else"
    doc = one_json_doc(p.stdout)
    assert doc["status"] == "failed" and doc["reason"]
    assert toy.artifacts() == []


@pytest.mark.parametrize("argv_change", [
    ("drop", "--build-id"), ("drop", "--attempt"), ("drop", "--dataset"), ("drop", "--n-shards"), ("drop", "--shard"),
    ("set", "--attempt", "x"), ("set", "--threads", "two"), ("set", "--shard", "1.5"), ("add", "--no-such-flag"),
])
def test_xcheck_build_bad_arguments_exit_2(toy, argv_change):
    argv = toy.argv()
    if argv_change[0] == "drop":
        i = argv.index(argv_change[1])
        del argv[i:i + 2]
    elif argv_change[0] == "set":
        argv[argv.index(argv_change[1]) + 1] = argv_change[2]
    else:
        argv.append(argv_change[1])
    p = subprocess.run(argv, capture_output=True, text=True, env=toy.env(), timeout=60)
    print(f"{argv_change}: exit {p.returncode}; stdout {p.stdout!r}; stderr last line {p.stderr.strip().splitlines()[-1]!r}")
    assert p.returncode == 2, "decision 17: 2 usage"
    assert toy.artifacts() == []


@pytest.mark.parametrize("case", ["recall_k_beyond_truth", "recall_min_not_a_number", "threads_zero", "attempt_zero", "negative_shard"])
def test_xcheck_build_questionable_arguments(toy, case):
    """Values that parse but make no sense. The spec does not list them; the
    contract (decision 17) is: stdout ONE JSON document, exit 2 usage / 1
    anything else, and nothing published unless built and gated."""
    env, kw = toy.env(), {}
    if case == "recall_k_beyond_truth":
        kw = dict(extra=["--recall-k", "200"])  # the stored truth is top-100
    elif case == "recall_min_not_a_number":
        env = toy.env(RECALL_MIN="abc")
    elif case == "threads_zero":
        kw = dict(threads=0)
    elif case == "attempt_zero":
        kw = dict(attempt=0)
    else:
        kw = dict(shard=-1)
    p = toy.run("0:60", **kw, env=env)
    print(f"{case}: exit {p.returncode}; stdout {p.stdout.strip()[:160]!r}; stderr last line {(p.stderr.strip().splitlines() or [''])[-1]!r}; artifacts {toy.artifacts()}")
    if p.returncode == 0:
        print(f"  NOTE: {case} was accepted and published")
        return
    assert p.returncode in (1, 2)
    one_json_doc(p.stdout) if p.returncode == 1 else None  # exit 1 must still carry the failure document
    assert not any(k.endswith("manifest.json") for k in toy.artifacts())
