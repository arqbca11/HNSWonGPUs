"""Cost model v1, mirroring scheduler/costmodel/costmodel.go (Phase 5 decision 6).

The worker needs GPU total time (what the fake build sleeps for, and the
shutdown decision). The formulas and defaults must stay identical to the Go
package; test_costmodel.py pins the Go test's numbers.

    CPU build   = A · n^P · ln(n) · (dim + D0)
    gate        = Gate · GateQueries · ln(n) · (dim + D0)
    save        = index bytes / SaveBandwidth            (0 when SaveBandwidth is 0)
    fetch       = vector bytes / Bandwidth
    upload      = index bytes / BandwidthOut             (Bandwidth when BandwidthOut is 0)
    GPU build   = (CPU build + gate) / Speedup + GPUOverhead + save
    transfer    = fetch + upload
    GPU total   = GPU build + transfer
    local total = CPU build + gate + save [+ transfer if shard_uses_store]
    vector bytes = n · dim · 4;  index bytes = n · (dim·4 + 2·M·4);  GPU memory = MemFactor · vector bytes

Model() is v0 (P 1, D0 0, no gate, no save, one bandwidth): the placeholder
the Phase 1-3 tests were written against. Model.calibrated() is the cloud
topology's measured block.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace


def _parse_bool(v: str) -> bool:
    """Go's strconv.ParseBool, so both sides accept the same spellings (G12)."""
    if v in ("1", "t", "T", "TRUE", "true", "True"):
        return True
    if v in ("0", "f", "F", "FALSE", "false", "False"):
        return False
    raise ValueError(f"COST_SHARD_USES_STORE must be true or false, got {v!r}")


@dataclass(frozen=True)
class Model:
    a: float = 1.4e-7
    p: float = 1.0
    d0: float = 0.0
    speedup: float = 10.0
    gpu_overhead: float = 2.0
    gate: float = 0.0
    gate_queries: float = 0.0
    save_bandwidth: float = 0.0
    bandwidth: float = 200e6
    bandwidth_out: float = 0.0
    shard_uses_store: bool = False
    m: int = 16
    mem_factor: float = 2.0
    bytes_per_dim: int = 4

    @classmethod
    def calibrated(cls) -> "Model":
        """The cloud topology's measured block; see costmodel.Calibrated() in Go."""
        return cls(a=2.30e-9, p=1.220, d0=187, speedup=2.12, gpu_overhead=1.5, gate=1.79e-8, gate_queries=10000,
                   save_bandwidth=6.6e8, bandwidth=1.57e8, bandwidth_out=5.0e7, shard_uses_store=True)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Model":
        """Same COST_* variables the scheduler reads, so one env block serves both.
        COST_MODEL=calibrated starts from the measured block instead of v0."""
        env = os.environ if env is None else env
        name = env.get("COST_MODEL", "")
        if name not in ("", "default", "calibrated"):
            # exactly the scheduler's two spellings: one env block serves both
            # processes and must mean the same on each (step 1 cross-check E)
            raise ValueError(f"COST_MODEL must be default or calibrated, got {name!r}")
        d = cls.calibrated() if name == "calibrated" else cls()
        f = lambda key, default: float(env[key]) if env.get(key) else default  # noqa: E731
        return replace(
            d,
            a=f("COST_A", d.a),
            p=f("COST_P", d.p),
            d0=f("COST_D0", d.d0),
            speedup=f("COST_SPEEDUP", d.speedup),
            gpu_overhead=f("COST_GPU_OVERHEAD", d.gpu_overhead),
            gate=f("COST_GATE", d.gate),
            gate_queries=f("COST_GATE_QUERIES", d.gate_queries),
            save_bandwidth=f("COST_SAVE_BANDWIDTH", d.save_bandwidth),
            bandwidth=f("COST_BANDWIDTH", d.bandwidth),
            bandwidth_out=f("COST_BANDWIDTH_OUT", d.bandwidth_out),
            shard_uses_store=_parse_bool(env["COST_SHARD_USES_STORE"]) if env.get("COST_SHARD_USES_STORE") else d.shard_uses_store,
            mem_factor=f("COST_MEM_FACTOR", d.mem_factor),
        )

    def vector_bytes(self, n: int, dim: int) -> int:
        return n * dim * self.bytes_per_dim

    def index_bytes(self, n: int, dim: int) -> int:
        return n * (dim * self.bytes_per_dim + 2 * self.m * 4)

    def gpu_mem_bytes(self, n: int, dim: int) -> int:
        return math.ceil(self.mem_factor * self.vector_bytes(n, dim))

    def _shape(self, n: int, dim: int) -> float:
        return n ** (self.p or 1.0) * math.log(n) * (dim + self.d0)

    def cpu_build_s(self, n: int, dim: int) -> float:
        if n < 2:
            return 0.0
        return self.a * self._shape(n, dim)

    def gate_s(self, n: int, dim: int) -> float:
        if n < 2 or self.gate == 0 or self.gate_queries == 0:
            return 0.0
        return self.gate * self.gate_queries * math.log(n) * (dim + self.d0)

    def save_s(self, n: int, dim: int) -> float:
        return self.index_bytes(n, dim) / self.save_bandwidth if self.save_bandwidth > 0 else 0.0

    def fetch_s(self, n: int, dim: int) -> float:
        return self.vector_bytes(n, dim) / self.bandwidth

    def upload_s(self, n: int, dim: int) -> float:
        return self.index_bytes(n, dim) / (self.bandwidth_out if self.bandwidth_out > 0 else self.bandwidth)

    def gpu_build_s(self, n: int, dim: int) -> float:
        return (self.cpu_build_s(n, dim) + self.gate_s(n, dim)) / self.speedup + self.gpu_overhead + self.save_s(n, dim)

    def transfer_s(self, n: int, dim: int) -> float:
        return self.fetch_s(n, dim) + self.upload_s(n, dim)

    def gpu_total_s(self, n: int, dim: int) -> float:
        """A worker's whole path: what a fake worker sleeps for."""
        return self.gpu_build_s(n, dim) + self.transfer_s(n, dim)

    def local_total_s(self, n: int, dim: int) -> float:
        d = self.cpu_build_s(n, dim) + self.gate_s(n, dim) + self.save_s(n, dim)
        return d + self.transfer_s(n, dim) if self.shard_uses_store else d
