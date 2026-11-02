"""The Python cost model must agree with the Go one. These numbers are the Go
test's output (scheduler/costmodel/costmodel_test.go, defaults)."""

from kgpu_worker.costmodel import Model


def test_matches_go_defaults():
    m = Model()
    cpu = m.cpu_build_s(100_000, 128)
    gpu = m.gpu_total_s(100_000, 128)
    mem = m.gpu_mem_bytes(100_000, 128)
    print(f"100k x 128: cpu={cpu:.3f}s gpu_total={gpu:.3f}s mem={mem}")
    # Go: cpu=20.631162433s, gpu(total incl. transfer)=4.639116243s, mem_bytes=102400000
    assert abs(cpu - 20.631162433) < 1e-6
    assert abs(gpu - 4.639116243) < 1e-6
    assert mem == 102_400_000
    # Go: 1k x 128: cpu=123.786974ms gpu=2.018138697s
    assert abs(m.cpu_build_s(1_000, 128) - 0.123786974) < 1e-6
    assert abs(m.gpu_total_s(1_000, 128) - 2.018138697) < 1e-6


def test_env_overrides():
    m = Model.from_env({"COST_SPEEDUP": "20", "COST_GPU_OVERHEAD": "0"})
    assert m.speedup == 20 and m.gpu_overhead == 0
    assert m.gpu_build_s(100_000, 128) == m.cpu_build_s(100_000, 128) / 20


def test_v1_default_is_v0_and_calibrated_matches_go():
    """Phase 5 decision 6: v1's formulas with v0's constants are v0 to the digit,
    and the calibrated block reproduces the Go test's hand-computed terms
    (scheduler/costmodel/costmodel_test.go, TestCalibratedTermsByHand)."""
    m = Model()
    assert m.local_total_s(100_000, 128) == m.cpu_build_s(100_000, 128)
    assert m.gate_s(100_000, 128) == 0 and m.save_s(100_000, 128) == 0
    c = Model.calibrated()
    n, dim = 408_164, 128
    terms = {
        "build": (c.cpu_build_s(n, dim), 65.539), "gate": (c.gate_s(n, dim), 0.728), "save": (c.save_s(n, dim), 0.396),
        "fetch": (c.fetch_s(n, dim), 1.331), "upload": (c.upload_s(n, dim), 5.224),
        "gpu build": (c.gpu_build_s(n, dim), 33.154), "transfer": (c.transfer_s(n, dim), 6.556),
        "gpu total": (c.gpu_total_s(n, dim), 39.710), "local total": (c.local_total_s(n, dim), 73.219),
    }
    for k, (got, want) in terms.items():
        print(f"calibrated 408164 x 128: {k:12s} = {got:8.3f} s (Go: {want})")
        assert abs(got - want) < 2e-3, k
    # the dimension term: well under v0's 7.5x for 960 over 128 dims (Phase 4 decision 35)
    r = c.cpu_build_s(10_000, 960) / c.cpu_build_s(10_000, 128)
    assert 2.5 < r < 4.5


def test_v1_env_block():
    m = Model.from_env({"COST_MODEL": "calibrated", "COST_SPEEDUP": "3", "COST_BANDWIDTH_OUT": "0", "COST_SHARD_USES_STORE": "false"})
    assert m.p == 1.220 and m.d0 == 187 and m.speedup == 3
    assert m.upload_s(1000, 128) == m.index_bytes(1000, 128) / m.bandwidth   # out = in when 0
    assert m.local_total_s(1000, 128) == m.cpu_build_s(1000, 128) + m.gate_s(1000, 128) + m.save_s(1000, 128)
    assert Model.from_env({}).p == 1.0 and Model.from_env({}).shard_uses_store is False


def test_v2_l4_block_matches_go():
    """Phase 5 decision 14: the L4 block gives the worker its own shape; the
    numbers are the Go test's (TestCalibratedL4TermsByHand)."""
    m = Model.calibrated_l4()
    n, dim = 408_164, 128
    for k, got, want in [("worker build + gate", m.worker_build_s(n, dim), 6.618), ("gpu build", m.gpu_build_s(n, dim), 9.019),
                         ("transfer", m.transfer_s(n, dim), 4.090), ("gpu total", m.gpu_total_s(n, dim), 13.109),
                         ("local total", m.local_total_s(n, dim), 71.258)]:
        print(f"L4 408164 x 128: {k:20s} = {got:8.3f} s (Go: {want})")
        assert abs(got - want) < 2e-3, k
    assert m.worker_build_s(10_000, 128) > m.cpu_build_s(10_000, 128)   # the card's floor: slower than the shard at 10k
    assert m.worker_build_s(1, 128) == 1.154
    v1 = Model.calibrated()
    assert v1.worker_build_s(n, dim) == 0.0 and v1.worker_a == 0
    e = Model.from_env({"COST_MODEL": "calibrated-l4", "COST_WORKER_FIXED": "2"})
    assert e.worker_fixed == 2 and e.worker_a == 1.295e-8
