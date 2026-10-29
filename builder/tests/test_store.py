"""The object store against a throwaway MinIO (scripts/test-db.sh starts one
and sets TEST_S3_ENDPOINT); skipped without it."""

import os

import numpy as np
import pytest

from kgpu_builder.config import Config
from kgpu_builder.store import Store


@pytest.fixture
def store():
    ep = os.environ.get("TEST_S3_ENDPOINT")
    if not ep:
        pytest.skip("TEST_S3_ENDPOINT not set; run scripts/test-db.sh")
    cfg = Config.from_env({
        "S3_ENDPOINT": ep, "S3_ACCESS_KEY": os.environ["TEST_S3_ACCESS_KEY"],
        "S3_SECRET_KEY": os.environ["TEST_S3_SECRET_KEY"], "S3_BUCKET": "kgpu-test",
    })
    st = Store(cfg)
    st.ensure_bucket()
    st.delete_prefix("")
    return st


def test_bytes_json_npy_round_trip_and_listing(store):
    store.put_bytes("a/b.bin", b"\x00\x01\x02")
    store.put_json("a/m.json", {"k": [1, 2]})
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    n = store.put_npy("a/x.npy", a)
    assert store.get_bytes("a/b.bin") == b"\x00\x01\x02"
    assert store.get_json("a/m.json") == {"k": [1, 2]}
    np.testing.assert_array_equal(store.get_npy("a/x.npy"), a)
    assert store.size("a/x.npy") == n
    assert sorted(store.list("a/")) == ["a/b.bin", "a/m.json", "a/x.npy"]
    assert store.exists("a/x.npy") and not store.exists("a/nope")
    assert store.delete_prefix("a/") == 3 and store.list("") == []
    print("bytes, JSON and .npy round-trip through MinIO; listing, exists and delete_prefix behave")


def test_ensure_bucket_is_idempotent(store):
    store.ensure_bucket()
    store.ensure_bucket()
