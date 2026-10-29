"""Cross-check (black box): the object-store wrapper against a real RustFS.

Spec: Phase 4 decisions 1, 9, 11, 13; the store module's docstring (put/get
bytes, JSON and .npy, keyed by object name, in one bucket; the object
layout, including artifacts/<build_id>/<attempt>/ where build_id is
'shard_id:round_id', invariant 6). Skips unless TEST_S3_ENDPOINT is set.
Every test uses its own fresh bucket and empties it at the end.
"""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from kgpu_builder.config import Config
from kgpu_builder.store import Store

S3 = os.environ.get("TEST_S3_ENDPOINT")
pytestmark = pytest.mark.skipif(not S3, reason="TEST_S3_ENDPOINT unset")


@pytest.fixture
def st(tmp_path):
    bucket = "xcheck-" + uuid.uuid4().hex[:10]
    cfg = Config(
        data_dir=Path(tmp_path), s3_endpoint=S3,
        s3_access_key=os.environ.get("TEST_S3_ACCESS_KEY", ""),
        s3_secret_key=os.environ.get("TEST_S3_SECRET_KEY", ""),
        s3_bucket=bucket, s3_region="us-east-1",
    )
    s = Store(cfg)
    s.ensure_bucket()
    print(f"(fresh bucket {bucket} on {S3})")
    yield s
    s.delete_prefix("")


def test_xcheck_store_large_object_round_trip(st):
    rng = np.random.default_rng(0)
    data = rng.bytes(48 * 1024 * 1024 + 123)  # past the usual 8 MB multipart threshold
    st.put_bytes("big/blob.bin", data)
    back = st.get_bytes("big/blob.bin")
    print(f"did: put 48 MiB + 123 bytes of random data; size() says {st.size('big/blob.bin')}; got back {len(back)} bytes; equal: {back == data}")
    assert back == data and st.size("big/blob.bin") == len(data)
    a = rng.standard_normal((100_000, 128)).astype(np.float32)  # 51 MB, a mid-size shard
    n = st.put_npy("big/shard.npy", a)
    b = st.get_npy("big/shard.npy")
    print(f"did: put_npy of a 100,000 x 128 float32 array -> returned {n}; size() {st.size('big/shard.npy')}; get_npy equal: {np.array_equal(a, b)}, dtype {b.dtype}")
    assert np.array_equal(a, b) and b.dtype == np.float32
    assert n == st.size("big/shard.npy"), "put_npy's int is the object's size (spec: returns int, unspecified; this is the natural reading)"


def test_xcheck_store_listing_and_deleting_beyond_1000_keys(st):
    """S3 pages ListObjects at 1000 keys and DeleteObjects takes at most
    1000 keys per call; the wrapper must hide both."""
    keys = [f"many/k{i:05d}" for i in range(2345)]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda k: st.put_bytes(k, k.encode()), keys))
    listed = st.list("many/")
    print(f"did: put 2345 small objects under many/\n  list('many/') returned {len(listed)} keys (expected 2345, sorted set equal: {sorted(listed) == keys})")
    assert sorted(listed) == keys
    n = st.delete_prefix("many/")
    after = st.list("many/")
    print(f"did: delete_prefix('many/') -> returned {n}; keys left {len(after)} (expected 2345 and 0)")
    assert n == 2345 and after == []


def test_xcheck_store_overwrite_replaces(st):
    st.put_bytes("o/x", b"first version, longer")
    st.put_bytes("o/x", b"second")
    print(f"did: put o/x twice; get -> {st.get_bytes('o/x')!r}; size {st.size('o/x')}; list {st.list('o/')}")
    assert st.get_bytes("o/x") == b"second" and st.size("o/x") == 6 and st.list("o/") == ["o/x"]
    st.put_json("o/m.json", {"a": 1})
    st.put_json("o/m.json", {"b": [1, 2], "é": "ü"})
    assert st.get_json("o/m.json") == {"b": [1, 2], "é": "ü"}


def test_xcheck_store_missing_key(st):
    print(f"did: exists('nope/x') -> {st.exists('nope/x')} (expected False)")
    assert st.exists("nope/x") is False
    for fn in (st.get_bytes, st.get_json, st.get_npy, st.size):
        with pytest.raises(Exception) as ei:
            fn("nope/x")
        print(f"  {fn.__name__}('nope/x') raised {type(ei.value).__module__}.{type(ei.value).__name__}: {str(ei.value)[:90]}")


def test_xcheck_store_delete_prefix_empty_on_empty_bucket(st):
    n = st.delete_prefix("")
    print(f"did: delete_prefix('') on an empty bucket -> {n} (expected 0, no error); list('') -> {st.list('')}")
    assert n == 0 and st.list("") == []
    assert st.delete_prefix("does/not/exist/") == 0


def test_xcheck_store_prefix_is_a_string_prefix(st):
    """splits/<ds>/5 is a string prefix of splits/<ds>/50. The spec does not
    say whether list/delete_prefix treat the prefix as a string or a
    directory; record it, because delete_prefix('splits/sift-128/5') would
    then also delete the 50-shard split."""
    for k in ("p/5/a", "p/50/a", "p/5x"):
        st.put_bytes(k, b"1")
    listed = sorted(st.list("p/5"))
    n = st.delete_prefix("p/5")
    left = sorted(st.list("p/"))
    print(f"did: keys p/5/a, p/50/a, p/5x\n  list('p/5') -> {listed}\n  delete_prefix('p/5') -> {n}; left {left}\n"
          f"  (string-prefix semantics: callers must pass a trailing '/')")
    assert n == len(listed)


def test_xcheck_store_artifact_keys_with_a_colon(st):
    """build_id is 'shard_id:round_id' (invariant 6) and artifacts live under
    artifacts/<build_id>/<attempt>/ (decision 9). A colon in a key must
    round-trip and list exactly."""
    k1 = "artifacts/3:17/1/index.bin"
    k2 = "artifacts/3:17/2/manifest.json"
    st.put_bytes(k1, b"index")
    st.put_json(k2, {"build_id": "3:17", "attempt": 2})
    listed = sorted(st.list("artifacts/3:17/"))
    print(f"did: put {k1} and {k2}; list -> {listed}; get -> {st.get_bytes(k1)!r}, {st.get_json(k2)}")
    assert listed == [k1, k2]
    assert st.get_bytes(k1) == b"index" and st.get_json(k2)["attempt"] == 2
    assert st.exists(k1) and not st.exists("artifacts/3:17/3/index.bin")


def test_xcheck_store_npy_dtypes_and_shapes(st):
    cases = {
        "i32": np.arange(12, dtype=np.int32).reshape(3, 4),
        "f32": np.ones((0, 128), np.float32),
        "f64": np.array(3.5),
        "u8": np.arange(256, dtype=np.uint8),
        "fortran": np.asfortranarray(np.arange(6, dtype=np.float32).reshape(2, 3)),
    }
    for name, a in cases.items():
        st.put_npy(f"npy/{name}.npy", a)
        b = st.get_npy(f"npy/{name}.npy")
        print(f"  {name}: put shape {a.shape} {a.dtype} -> got {b.shape} {b.dtype}, equal {np.array_equal(a, b)}")
        assert b.dtype == a.dtype and b.shape == a.shape and np.array_equal(a, b)


def test_xcheck_store_wrong_credentials_fail(tmp_path, st):
    bad = Store(Config(data_dir=Path(tmp_path), s3_endpoint=S3, s3_access_key="nobody",
                       s3_secret_key="wrong-secret", s3_bucket="kgpu", s3_region="us-east-1"))
    with pytest.raises(Exception) as ei:
        bad.put_bytes("x", b"1")
    print(f"did: put with wrong keys -> {type(ei.value).__name__}: {str(ei.value)[:100]}")
