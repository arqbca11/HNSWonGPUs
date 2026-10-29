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
import os
from typing import Any

import boto3
import numpy as np
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from .config import Config


class NotFound(KeyError):
    """The object does not exist. Raised by every getter and by size(), so
    callers never catch botocore types (cross-check G6)."""


def _is_not_found(e: ClientError) -> bool:
    return e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound")


def _dir(prefix: str) -> str:
    """A prefix names a directory: "splits/sift-128/5" must not also match
    "splits/sift-128/50" (cross-check G7), so a non-empty prefix without a
    trailing slash gets one. "" means the whole bucket."""
    return prefix if prefix == "" or prefix.endswith("/") else prefix + "/"


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

    def put_file(self, name: str, path: str) -> int:
        """Uploads a file from disk in parts (boto3's managed transfer), so a
        large index never has to be held in memory as one blob; returns its
        size in bytes."""
        self.s3.upload_file(path, self.bucket, name)
        return os.path.getsize(path)

    def get_bytes(self, name: str) -> bytes:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=name)["Body"].read()
        except ClientError as e:
            if _is_not_found(e):
                raise NotFound(name) from None
            raise

    def exists(self, name: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=name)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def size(self, name: str) -> int:
        try:
            return int(self.s3.head_object(Bucket=self.bucket, Key=name)["ContentLength"])
        except ClientError as e:
            if _is_not_found(e):
                raise NotFound(name) from None
            raise

    def put_json(self, name: str, obj: Any) -> None:
        self.put_bytes(name, json.dumps(obj, indent=2).encode(), "application/json")

    def get_json(self, name: str) -> Any:
        return json.loads(self.get_bytes(name))

    def put_npy(self, name: str, a: np.ndarray) -> int:
        """Stores the array in .npy form; returns the object's size in bytes."""
        buf = io.BytesIO()
        np.save(buf, a, allow_pickle=False)
        b = buf.getvalue()
        self.put_bytes(name, b)
        return len(b)

    def get_npy(self, name: str) -> np.ndarray:
        return np.load(io.BytesIO(self.get_bytes(name)), allow_pickle=False)

    def list(self, prefix: str) -> list[str]:
        """Every key under the directory `prefix` ("" = the whole bucket)."""
        out: list[str] = []
        token = None
        prefix = _dir(prefix)
        while True:
            kw = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            r = self.s3.list_objects_v2(**kw)
            out += [o["Key"] for o in r.get("Contents", [])]
            if not r.get("IsTruncated"):
                return out
            token = r.get("NextContinuationToken")

    def delete(self, name: str) -> None:
        """Deletes one object by its exact key (no error if it is absent)."""
        self.s3.delete_object(Bucket=self.bucket, Key=name)

    def delete_prefix(self, prefix: str) -> int:
        """Deletes every key under the directory `prefix`; returns how many."""
        keys = self.list(prefix)
        for i in range(0, len(keys), 1000):
            self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": [{"Key": k} for k in keys[i:i + 1000]]})
        return len(keys)
