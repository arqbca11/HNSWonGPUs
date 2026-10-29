"""The build step behind one interface (invariant 9).

    Builder.build(job, progress, cancel) -> Artifact

Fake (this file), hnswlib (Phase 4) and cuVS (Phase 5) are implementations.
Nothing else in the worker changes when one is added.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from .costmodel import Model


@dataclass(frozen=True)
class Job:
    build_id: str
    round_id: int
    attempt: int
    n_vectors: int
    dim: int
    mem_bytes: int
    dataset: str | None = None  # Phase 4 decision 15: the vectors are splits/<dataset>/<n_shards>/vectors/shard_<shard>.npy
    n_shards: int = 0

    @property
    def shard_id(self) -> int:
        return int(self.build_id.split(":")[0])


@dataclass(frozen=True)
class Artifact:
    build_id: str
    attempt: int
    kind: str          # "fake", later "hnswlib", "cagra-hnsw"
    index_bytes: int   # modeled or real size of what was produced
    elapsed_s: float


class Cancelled(Exception):
    """The build was abandoned: lease lost, or shutdown."""


class Builder(Protocol):
    name: str

    def build(self, job: Job, progress: Callable[[float], None], cancel: threading.Event) -> Artifact: ...

    def estimate_s(self, job: Job) -> float:
        """How long build() is expected to take; used for the shutdown decision."""
        ...


class FakeBuilder:
    """Sleeps for the cost model's GPU time plus transfer time, in slices, so it
    can report progress and notice cancellation. time_scale > 1 speeds up
    everything (tests, demos) without touching the model."""

    name = "fake"

    def __init__(self, model: Model, time_scale: float = 1.0, slice_s: float = 0.1,
                 fail_build_ids: frozenset[str] = frozenset()) -> None:
        if time_scale <= 0:
            raise ValueError("time_scale must be positive")
        self.model = model
        self.time_scale = time_scale
        self.slice_s = slice_s
        self.fail_build_ids = fail_build_ids   # these fail at 50%, to exercise the fail path

    def estimate_s(self, job: Job) -> float:
        return self.model.gpu_total_s(job.n_vectors, job.dim) / self.time_scale

    def build(self, job: Job, progress: Callable[[float], None], cancel: threading.Event) -> Artifact:
        total = self.estimate_s(job)
        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            # Injected failure first, so it fires even for builds shorter than one slice.
            if job.build_id in self.fail_build_ids and elapsed >= total / 2:
                raise RuntimeError(f"injected failure for {job.build_id} (FAKE_FAIL_BUILD_IDS)")
            if elapsed >= total:
                break
            if cancel.is_set():
                raise Cancelled(f"build {job.build_id} attempt {job.attempt} cancelled at {elapsed / total:.0%}")
            progress(min(elapsed / total, 1.0))
            time.sleep(min(self.slice_s, total - elapsed))
        progress(1.0)
        return Artifact(
            build_id=job.build_id,
            attempt=job.attempt,
            kind=self.name,
            index_bytes=self.model.index_bytes(job.n_vectors, job.dim),
            elapsed_s=time.monotonic() - start,
        )


class GateRefused(Exception):
    """The builder's recall gate refused the index (exit 3): the build is
    marked failed with this reason, like any other build failure."""


class HnswlibBuilder:
    """Runs `python -m kgpu_builder build ...` as a child process (Phase 4
    decisions 3, 17, 18): the same program the Go shard runs for a local
    build. stderr progress lines feed `progress`; `cancel` kills the child
    (the lease is gone, so nothing it could still write may be completed);
    exit 3 becomes GateRefused with the gate's reason; exit 0's manifest on
    stdout becomes the Artifact. The object store's address and keys reach
    the child through the environment (S3_*), as does RECALL_MIN."""

    name = "hnswlib"

    def __init__(self, model: Model, cmd: list[str], threads: int = 8, env: dict[str, str] | None = None) -> None:
        self.model = model
        self.cmd = cmd
        self.threads = threads
        self.env = env

    def estimate_s(self, job: Job) -> float:
        # the cost model's estimate until Phase 4's calibration replaces its constants
        return self.model.gpu_total_s(job.n_vectors, job.dim)

    def build(self, job: Job, progress: Callable[[float], None], cancel: threading.Event) -> Artifact:
        if not job.dataset:
            raise RuntimeError(f"{job.build_id}: a modeled round (no dataset) cannot be built by hnswlib; run BUILDER=fake")
        import json
        import os
        import subprocess
        argv = self.cmd + ["build", "--build-id", job.build_id, "--attempt", str(job.attempt), "--dataset", job.dataset,
                           "--n-shards", str(job.n_shards), "--shard", str(job.shard_id), "--threads", str(self.threads)]
        start = time.monotonic()
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                env={**os.environ, **(self.env or {})}, start_new_session=True)
        out_lines: list[str] = []

        def pump_stdout() -> None:
            assert proc.stdout
            for line in proc.stdout:
                out_lines.append(line)

        def pump_stderr() -> None:
            # progress lines as they come; a blocked readline must never delay
            # the cancel check below, so this runs in its own thread
            assert proc.stderr
            for line in proc.stderr:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if "progress" in msg:
                    progress(float(msg["progress"]))

        pumps = [threading.Thread(target=pump_stdout, daemon=True), threading.Thread(target=pump_stderr, daemon=True)]
        for t in pumps:
            t.start()
        while proc.poll() is None:
            if cancel.is_set():
                try:
                    os.killpg(proc.pid, 9)   # the whole process group, like the shard's preempt
                except ProcessLookupError:
                    pass
                proc.wait()
                raise Cancelled(f"build {job.build_id} attempt {job.attempt} cancelled: child killed")
            time.sleep(0.05)
        rc = proc.returncode
        for t in pumps:
            t.join(timeout=5)
        out = "".join(out_lines).strip()
        if rc == 3:
            try:
                reason = json.loads(out).get("reason") or out
            except ValueError:
                reason = out
            raise GateRefused(reason or "the builder's gate refused the build and gave no reason")
        if rc != 0:
            raise RuntimeError(f"builder exited {rc}: {out[:300]}")
        m = json.loads(out)
        progress(1.0)
        return Artifact(build_id=job.build_id, attempt=job.attempt, kind=self.name,
                        index_bytes=int(m["index_bytes"]), elapsed_s=time.monotonic() - start)


def by_name(name: str, model: Model, time_scale: float, fail_build_ids: frozenset[str] = frozenset(),
            cmd: list[str] | None = None, threads: int = 8) -> Builder:
    if name in ("", "fake"):
        return FakeBuilder(model, time_scale=time_scale, fail_build_ids=fail_build_ids)
    if name == "hnswlib":
        import shlex
        import sys
        return HnswlibBuilder(model, cmd or [sys.executable, "-m", "kgpu_builder"], threads=threads)
    raise ValueError(f"unknown builder {name!r}")
