"""The object store, behind one small class: put/get bytes, JSON and .npy
arrays, keyed by object name, in one bucket. boto3 against MinIO today and
against any S3-compatible store in Phase 5 (decision 11).

Object layout (Phase 4 decision 9):
    datasets/<dataset>/queries.npy              the held-out queries
    datasets/<dataset>/manifest.json            dim, metric, counts, sha256
    splits/<dataset>/<n_shards>/manifest.json   the split (sizes, ranks, offsets, sha256 per object)
    splits/<dataset>/<n_shards>/vectors/shard_k.npy
    splits/<dataset>/<n_shards>/ground_truth/shard_k.npy   int32 [n_queries, 100], local indices
    artifacts/<build_id>/<attempt>/index.bin, manifest.json   (step 2; invariant 7)
"""

from __future__ import annotations

import io
import json
from typing import Any

import boto3
import numpy as np
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from .config import Config


class Store:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.bucket = cfg.s3_bucket
        self.s3 = boto3.client(
            "s3",
            endpoint_url=cfg.s3_endpoint,
            aws_access_key_id=cfg.s3_access_key,
            aws_secret_access_key=cfg.s3_secret_key,
            region_name=cfg.s3_region,
            config=BotoConfig(s3={"addressing_style": "path"}, retries={"max_attempts": 5}),
        )

    def ensure_bucket(self) -> None:
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") not in ("404", "NoSuchBucket"):
                raise
            self.s3.create_bucket(Bucket=self.bucket)

    def put_bytes(self, name: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self.s3.put_object(Bucket=self.bucket, Key=name, Body=data, ContentType=content_type)

    def get_bytes(self, name: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=name)["Body"].read()

    def exists(self, name: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=name)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def size(self, name: str) -> int:
        return int(self.s3.head_object(Bucket=self.bucket, Key=name)["ContentLength"])

    def put_json(self, name: str, obj: Any) -> None:
        self.put_bytes(name, json.dumps(obj, indent=2).encode(), "application/json")

    def get_json(self, name: str) -> Any:
        return json.loads(self.get_bytes(name))

    def put_npy(self, name: str, a: np.ndarray) -> int:
        buf = io.BytesIO()
        np.save(buf, a, allow_pickle=False)
        b = buf.getvalue()
        self.put_bytes(name, b)
        return len(b)

    def get_npy(self, name: str) -> np.ndarray:
        return np.load(io.BytesIO(self.get_bytes(name)), allow_pickle=False)

    def list(self, prefix: str) -> list[str]:
        out: list[str] = []
        token = None
        while True:
            kw = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            r = self.s3.list_objects_v2(**kw)
            out += [o["Key"] for o in r.get("Contents", [])]
            if not r.get("IsTruncated"):
                return out
            token = r.get("NextContinuationToken")

    def delete_prefix(self, prefix: str) -> int:
        keys = self.list(prefix)
        for i in range(0, len(keys), 1000):
            self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": [{"Key": k} for k in keys[i:i + 1000]]})
        return len(keys)
