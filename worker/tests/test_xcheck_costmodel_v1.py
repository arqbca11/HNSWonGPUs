"""Cross-check (black box) of Phase 5 step 1 on the worker: the cost model v1 mirror.

Spec only: Phase 5 decisions 6 and 7 and "Step 1 (GPU-free): cost model v1"
(docs/design/phase5-gpu.md), the module docstring of kgpu_worker.costmodel and
its public signatures, `go doc ./scheduler/costmodel`. The implementation
(costmodel.py, test_costmodel.py) was NOT read. Go-vs-Python agreement on every
term is checked from the Go side (scheduler/crosscheck/p5s1_costmodel_test.go,
TestXcP5S1PythonMirrorAgrees); here the mirror is checked against the formulas
by hand and its environment block is exercised name by name.
"""

from __future__ import annotations

import math

import pytest

from kgpu_worker.costmodel import Model

V1_NAMES = {
    "COST_A": ("a", "4e-9", 4e-9),
    "COST_P": ("p", "1.1", 1.1),
    "COST_D0": ("d0", "99", 99.0),
    "COST_SPEEDUP": ("speedup", "3", 3.0),
    "COST_GPU_OVERHEAD": ("gpu_overhead", "2.5", 2.5),
    "COST_GATE": ("gate", "1e-8", 1e-8),
    "COST_GATE_QUERIES": ("gate_queries", "500", 500.0),
    "COST_SAVE_BANDWIDTH": ("save_bandwidth", "9e8", 9e8),
    "COST_BANDWIDTH": ("bandwidth", "1.2e8", 1.2e8),
    "COST_BANDWIDTH_OUT": ("bandwidth_out", "4e7", 4e7),
    "COST_SHARD_USES_STORE": ("shard_uses_store", "false", False),
    "COST_MEM_FACTOR": ("mem_factor", "3", 3.0),
}

CAL = dict(a=2.30e-9, p=1.220, d0=187.0, speedup=2.12, gpu_overhead=1.5, gate=1.79e-8, gate_queries=10000.0,
           save_bandwidth=6.6e8, bandwidth=1.57e8, bandwidth_out=5.0e7, shard_uses_store=True, m=16, bytes_per_dim=4)
V0 = dict(a=1.4e-7, p=1.0, d0=0.0, speedup=10.0, gpu_overhead=2.0, gate=0.0, gate_queries=0.0, save_bandwidth=0.0,
          bandwidth=2e8, bandwidth_out=0.0, shard_uses_store=False, m=16, bytes_per_dim=4)


def hand(m: Model, n: int, dim: int) -> dict[str, float]:
    """Decision 6, by hand."""
    vec = n * dim * m.bytes_per_dim
    idx = n * (dim * m.bytes_per_dim + 2 * m.m * 4)
    cpu = m.a * n ** m.p * math.log(n) * (dim + m.d0) if n >= 2 else 0.0
    gate = m.gate * m.gate_queries * math.log(n) * (dim + m.d0) if n >= 2 else 0.0
    save = idx / m.save_bandwidth if m.save_bandwidth else 0.0
    fetch = vec / m.bandwidth
    upload = idx / (m.bandwidth_out or m.bandwidth)
    gpu_build = (cpu + gate) / m.speedup + m.gpu_overhead + save
    transfer = fetch + upload
    local = cpu + gate + save + (transfer if m.shard_uses_store else 0.0)
    return dict(cpu=cpu, gate=gate, save=save, fetch=fetch, upload=upload, gpu_build=gpu_build, transfer=transfer,
                gpu_total=gpu_build + transfer, local=local, vec=vec, idx=idx, mem=m.mem_factor * vec)


def got(m: Model, n: int, dim: int) -> dict[str, float]:
    return dict(cpu=m.cpu_build_s(n, dim), gate=m.gate_s(n, dim), save=m.save_s(n, dim), fetch=m.fetch_s(n, dim),
                upload=m.upload_s(n, dim), gpu_build=m.gpu_build_s(n, dim), transfer=m.transfer_s(n, dim),
                gpu_total=m.gpu_total_s(n, dim), local=m.local_total_s(n, dim), vec=m.vector_bytes(n, dim),
                idx=m.index_bytes(n, dim), mem=m.gpu_mem_bytes(n, dim))


def mismatches(m: Model, label: str) -> list[str]:
    bad = []
    for n in (2, 3, 1000, 68027, 100000, 408164, 1_000_000, 10_000_000):
        for d in (1, 128, 960, 1536):
            g, h = got(m, n, d), hand(m, n, d)
            for k in g:
                tol = 1.0 if k in ("vec", "idx", "mem") else 1e-12 * abs(h[k]) + 1e-12
                if abs(g[k] - h[k]) > tol:
                    bad.append(f"{label} n={n} dim={d} {k}: got {g[k]!r}, hand {h[k]!r}")
    return bad


def fields(m: Model, keys) -> dict:
    return {k: getattr(m, k) for k in keys}


def test_xc_v0_default_to_the_digit():
    m = Model()
    print(f"Model() = {m}")
    print("spec (decision 7): Model() is v0; 100k x 128: cpu 20.631162433 s, gpu total 4.639116243 s")
    for k, v in V0.items():
        assert getattr(m, k) == v, f"Model().{k} = {getattr(m, k)!r}, v0 is {v!r}"
    cpu, tot = m.cpu_build_s(100000, 128), m.gpu_total_s(100000, 128)
    print(f"cpu_build_s {cpu:.9f}, gpu_total_s {tot:.9f}, local_total_s {m.local_total_s(100000, 128):.9f}, "
          f"gate {m.gate_s(100000, 128)}, save {m.save_s(100000, 128)}")
    assert round(cpu, 9) == 20.631162433
    assert round(tot, 9) == 4.639116243
    assert m.local_total_s(100000, 128) == cpu, "v0's local path is the build alone"
    assert m.gate_s(100000, 128) == 0 and m.save_s(100000, 128) == 0
    bad = mismatches(m, "Model()")
    print(f"=> every term of Model() matches decision 6's formula with v0's constants: {len(bad)} mismatches")
    assert not bad, bad[:10]


def test_xc_calibrated_is_the_block():
    m = Model.calibrated()
    print(f"Model.calibrated() = {m}")
    print("spec (decision 7; the Go Calibrated()): A 2.30e-9, P 1.220, D0 187, Speedup 2.12, GPUOverhead 1.5, Gate 1.79e-8, "
          "GateQueries 10000, SaveBandwidth 6.6e8, Bandwidth 1.57e8, BandwidthOut 5.0e7, ShardUsesStore true")
    for k, v in CAL.items():
        assert getattr(m, k) == pytest.approx(v, rel=1e-12), f"calibrated().{k} = {getattr(m, k)!r}, block says {v!r}"
    g = got(m, 408164, 128)
    print("408164 x 128: " + ", ".join(f"{k} {v:.3f}" for k, v in g.items() if k not in ("vec", "idx", "mem")))
    bad = mismatches(m, "calibrated()")
    print(f"=> {len(bad)} mismatches against the formulas by hand")
    assert not bad, bad[:10]


def test_xc_every_term_by_hand_custom_model():
    m = Model(a=3.1e-9, p=1.17, d0=55, speedup=3.3, gpu_overhead=0.7, gate=2.2e-8, gate_queries=1234,
              save_bandwidth=4.4e8, bandwidth=1.1e8, bandwidth_out=3.3e7, shard_uses_store=True, mem_factor=2.5)
    print(f"custom model, every term non-zero: {m}")
    bad = mismatches(m, "custom")
    for variant in (dict(bandwidth_out=0.0), dict(save_bandwidth=0.0), dict(shard_uses_store=False), dict(gate=0.0)):
        mv = Model(**{**fields(m, V0.keys()), "mem_factor": m.mem_factor, **variant})
        b = mismatches(mv, f"custom {variant}")
        print(f"  variant {variant}: {len(b)} mismatches; upload(250k,128) {mv.upload_s(250000, 128):.6f}, "
              f"save {mv.save_s(250000, 128):.6f}, local {mv.local_total_s(250000, 128):.6f}")
        bad += b
    print(f"=> {len(bad)} mismatches (BandwidthOut 0 falls back to Bandwidth; SaveBandwidth 0 means no save)")
    assert not bad, bad[:10]


def test_xc_tiny_builds_cost_nothing():
    print("spec (brief): a build of fewer than 2 vectors costs 0; which terms is not written down, so cpu and gate are asserted")
    for label, m in (("Model()", Model()), ("calibrated()", Model.calibrated())):
        for n in (0, 1):
            g = got(m, n, 128)
            print(f"{label} n={n}: " + ", ".join(f"{k} {v}" for k, v in g.items()))
            assert g["cpu"] == 0 and g["gate"] == 0
            assert all(math.isfinite(v) and v >= 0 for v in g.values())


def test_xc_from_env_empty_is_v0_and_calibrated_selects_the_block():
    e = Model.from_env({})
    c = Model.from_env({"COST_MODEL": "calibrated"})
    d = Model.from_env({"COST_MODEL": "default"})
    print(f"from_env({{}}) = {e}\nfrom_env(COST_MODEL=calibrated) = {c}\nfrom_env(COST_MODEL=default) = {d}")
    print("spec (decision 7): COST_MODEL=calibrated selects the block on the worker as on the scheduler; empty is v0")
    assert e == Model() and d == Model()
    assert c == Model.calibrated()


@pytest.mark.parametrize("start", ["default", "calibrated"])
@pytest.mark.parametrize("name", sorted(V1_NAMES))
def test_xc_from_env_each_override_name(start, name):
    attr, raw, val = V1_NAMES[name]
    base = Model() if start == "default" else Model.calibrated()
    m = Model.from_env({"COST_MODEL": start, name: raw})
    print(f"COST_MODEL={start} {name}={raw}: {attr} {getattr(base, attr)!r} -> {getattr(m, attr)!r}")
    print("spec (brief, decision 7): the COST_* names apply on top of the selected model, the same names the scheduler reads")
    assert getattr(m, attr) == pytest.approx(val) if not isinstance(val, bool) else getattr(m, attr) is val
    for k in V0:
        if k != attr:
            assert getattr(m, k) == getattr(base, k), f"{name} must change only {attr}; {k} moved"


def test_xc_from_env_bandwidth_out_zero_and_bool_spellings():
    m = Model.from_env({"COST_MODEL": "calibrated", "COST_BANDWIDTH_OUT": "0"})
    print(f"calibrated + COST_BANDWIDTH_OUT=0: bandwidth_out {m.bandwidth_out}, upload(408164,128) {m.upload_s(408164, 128):.4f} "
          f"= index/bandwidth {m.index_bytes(408164, 128) / m.bandwidth:.4f}")
    assert m.upload_s(408164, 128) == pytest.approx(m.index_bytes(408164, 128) / m.bandwidth)
    seen = {}
    for s in ("true", "false", "1", "0", "True", "FALSE", "yes", "no"):
        try:
            seen[s] = Model.from_env({"COST_SHARD_USES_STORE": s}).shard_uses_store
        except Exception as ex:  # noqa: BLE001 - recording what happens is the point
            seen[s] = f"raises {type(ex).__name__}"
    print(f"COST_SHARD_USES_STORE spellings -> {seen} (spec silent on accepted spellings)")
    assert seen["true"] is True and seen["false"] is False


def test_xc_from_env_unknown_model_name():
    print("spec: the scheduler exits 2 on any COST_MODEL other than default|calibrated; the worker's behaviour is not written down")
    for v in ("bogus", "Calibrated", "v1"):
        try:
            m = Model.from_env({"COST_MODEL": v})
            what = "calibrated block" if m == Model.calibrated() else ("v0" if m == Model() else f"other {m}")
            print(f"  COST_MODEL={v!r}: accepted -> {what}")
        except Exception as ex:  # noqa: BLE001
            print(f"  COST_MODEL={v!r}: raises {type(ex).__name__}: {ex}")
    with pytest.raises(Exception):
        Model.from_env({"COST_MODEL": "bogus"})
