"""Environment-variable configuration, read once.

    KGPU_DATA_DIR    local directory for downloaded datasets and split
                     outputs; default ./data (gitignored)
    S3_ENDPOINT      object store URL; default http://127.0.0.1:9000 (RustFS on
                     Compose, or a port-forward / NodePort 30900 to the cluster)
    S3_ACCESS_KEY    default kgpu
    S3_SECRET_KEY    default kgpu-secret
    S3_BUCKET        default kgpu
    S3_REGION        default us-east-1 (MinIO ignores it; a cloud store needs it)

Phase 5 changes S3_ENDPOINT and the keys and nothing else (Phase 4
decision 1: an S3 API means the cloud phase changes a URL).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_dir: Path
    s3_endpoint: str
    s3_access_key: str
    s3_secret_key: str
    s3_bucket: str
    s3_region: str

    @staticmethod
    def from_env(env: dict[str, str] | None = None) -> "Config":
        e = os.environ if env is None else env
        return Config(
            data_dir=Path(e.get("KGPU_DATA_DIR", "data")),
            s3_endpoint=e.get("S3_ENDPOINT", "http://127.0.0.1:9000"),
            s3_access_key=e.get("S3_ACCESS_KEY", "kgpu"),
            s3_secret_key=e.get("S3_SECRET_KEY", "kgpu-secret"),
            s3_bucket=e.get("S3_BUCKET", "kgpu"),
            s3_region=e.get("S3_REGION", "us-east-1"),
        )
