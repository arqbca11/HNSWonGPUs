# kgpu-builder

Phase 4's Python: the dataset tools (download, Zipf split across 6 or 50 shards, exact per-shard ground truth, upload to the object store) and, from step 2, the hnswlib builder CLI that the Python worker and the Go shard both run.

    cd builder && uv sync
    uv run python -m kgpu_builder all sift-128 6        # download, split, ground truth, upload, verify
    uv run python -m kgpu_builder all sift-128 50
    uv run python -m kgpu_builder --help

Environment: `KGPU_DATA_DIR` (default `./data`, gitignored), `S3_ENDPOINT` (default `http://127.0.0.1:9000`), `S3_ACCESS_KEY`/`S3_SECRET_KEY` (default `kgpu`/`kgpu-secret`), `S3_BUCKET` (default `kgpu`). Tests: `uv run pytest` (the store test needs `TEST_S3_ENDPOINT`, which `scripts/test-db.sh` provides).
