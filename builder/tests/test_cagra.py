"""The GPU build path's scaffold (Phase 5 decision 10). Without cuVS — every
laptop and kind run — the module says so in one line and the CLI fails a
cagra build fast, before any fetch; with cuVS (the GPU image on a GPU node)
the build runs, the converted file loads in stock hnswlib and the gate holds."""

import json
import subprocess
import sys

import numpy as np
import pytest

from kgpu_builder import cagra


def test_without_cuvs_the_path_refuses_in_one_line():
    if cagra.cuvs_available():
        pytest.skip("cuVS is installed here: the refusal path does not apply")
    print(f"cuvs_available() = False; cuvs_version() = {cagra.cuvs_version()!r}")
    with pytest.raises(cagra.CuvsUnavailable) as e:
        cagra.build_index_cagra(np.zeros((10, 8), dtype=np.float32), 16, 200, 2)
    print(f"build_index_cagra -> CuvsUnavailable: {e.value}")
    assert "cuVS is not installed" in str(e.value) and "\n" not in str(e.value)


def test_cli_cagra_build_fails_fast_without_cuvs(tmp_path):
    if cagra.cuvs_available():
        pytest.skip("cuVS is installed here")
    # no store is reachable at this endpoint: the check must come before the fetch
    env = {"S3_ENDPOINT": "http://127.0.0.1:1", "S3_ACCESS_KEY": "x", "S3_SECRET_KEY": "x", "S3_BUCKET": "kgpu", "PATH": "/usr/bin:/bin"}
    p = subprocess.run([sys.executable, "-m", "kgpu_builder", "build", "--build-id", "0:1", "--attempt", "0", "--dataset", "sift-128",
                        "--n-shards", "6", "--shard", "0", "--builder", "cagra"], capture_output=True, text=True, env=env, timeout=60)
    print(f"exit {p.returncode}; stdout {p.stdout.strip()[:200]}")
    assert p.returncode == 1
    doc = json.loads(p.stdout)
    assert doc["status"] == "failed" and doc["reason"].startswith("CuvsUnavailable") and doc["build_id"] == "0:1"
    assert "--builder" in subprocess.run([sys.executable, "-m", "kgpu_builder", "build", "--help"], capture_output=True, text=True).stdout


@pytest.mark.skipif(not cagra.cuvs_available(), reason="needs cuVS and a GPU (the GPU image on a GPU node)")
def test_cagra_build_loads_in_stock_hnswlib_and_passes_the_gate():
    from kgpu_builder.build import gate
    from kgpu_builder.groundtruth import exact_topk
    rng = np.random.default_rng(1)
    x = rng.standard_normal((20_000, 64)).astype(np.float32)
    q = rng.standard_normal((200, 64)).astype(np.float32)
    truth = exact_topk(q, x, k=10)
    idx = cagra.build_index_cagra(x, 16, 200, 4)
    r = gate(idx, q, truth, 10, 100, 4)
    print(f"cagra -> hnsw -> stock hnswlib: {idx.get_current_count()} vectors, recall@10 {r:.3f}")
    assert idx.get_current_count() == 20_000 and r >= 0.9
