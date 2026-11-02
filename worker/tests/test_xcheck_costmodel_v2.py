"""Cross-check (black box) of cost model v2 on the worker: the mirror of the worker's own shape.

Spec only: Phase 5 decisions 13 and 14 (docs/design/phase5-gpu.md), the module docstring and public
signatures of kgpu_worker.costmodel, `go doc ./scheduler/costmodel`, and the brief. costmodel.py and
test_costmodel.py were NOT read. Go-vs-Python agreement on every term (worker_build_s included) is checked
from the Go side (scheduler/crosscheck/p5v2_costmodel_test.go, TestXcP5V2PythonMirrorAgrees); here the mirror
is checked against the formulas by hand and its environment block is exercised name by name.

v2 by hand (decision 14): when worker_a > 0,
    worker build = worker_fixed + worker_a·n^worker_p·ln(n)·(dim+worker_d0) + worker_gate·gate_queries·ln(n)·(dim+worker_d0)
                   (n < 2: worker_fixed only)
    gpu build    = worker build + gpu_overhead + save
when worker_a = 0 the v1 form (cpu + gate)/speedup + gpu_overhead + save stands and worker_build_s is 0.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kgpu_worker.costmodel import Model

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_xcheck_costmodel_v1 as v1x  # noqa: E402  (the step 1 cross-check's hand model and blocks)

L4_FIELDS = dict(worker_fixed=1.154, worker_a=1.295e-8, worker_p=0.891, worker_d0=154.0, worker_gate=2.07e-8,
                 bandwidth=1.39e8, bandwidth_out=1.01e8, save_bandwidth=2.9e8)
WORKER_NAMES = {
    "COST_WORKER_FIXED": ("worker_fixed", "2.5", 2.5),
    "COST_WORKER_A": ("worker_a", "3e-8", 3e-8),
    "COST_WORKER_P": ("worker_p", "0.85", 0.85),
    "COST_WORKER_D0": ("worker_d0", "99", 99.0),
    "COST_WORKER_GATE": ("worker_gate", "1e-8", 1e-8),
}
ALL_FIELDS = list(v1x.V0) + ["mem_factor"] + [a for a, _, _ in WORKER_NAMES.values()]
SIZES = (0, 1, 2, 3, 1000, 68027, 100000, 408164, 1_000_000, 10_000_000)
DIMS = (1, 128, 960, 1536)


def hand_v2(m: Model, n: int, dim: int) -> dict[str, float]:
    t = v1x.hand(m, n, dim) if n >= 2 else _hand_tiny(m, n, dim)
    wb = 0.0
    if m.worker_a > 0:
        wb = m.worker_fixed
        if n >= 2:
            wb += (m.worker_a * n ** m.worker_p * math.log(n) * (dim + m.worker_d0)
                   + m.worker_gate * m.gate_queries * math.log(n) * (dim + m.worker_d0))
        t["gpu_build"] = wb + m.gpu_overhead + t["save"]
        t["gpu_total"] = t["gpu_build"] + t["transfer"]
    t["worker_build"] = wb
    return t


def _hand_tiny(m: Model, n: int, dim: int) -> dict[str, float]:
    # v1x.hand takes log(n) only when n >= 2 for cpu/gate; it is safe for n 0 and 1 too.
    return v1x.hand(m, n, dim)


def got_v2(m: Model, n: int, dim: int) -> dict[str, float]:
    g = v1x.got(m, n, dim)
    g["worker_build"] = m.worker_build_s(n, dim)
    return g


def mismatches(m: Model, label: str) -> list[str]:
    bad = []
    for n in SIZES:
        for d in DIMS:
            g, h = got_v2(m, n, d), hand_v2(m, n, d)
            for k in g:
                tol = 1.0 if k in ("vec", "idx", "mem") else 1e-12 * abs(h[k]) + 1e-12
                if abs(g[k] - h[k]) > tol:
                    bad.append(f"{label} n={n} dim={d} {k}: got {g[k]!r}, hand {h[k]!r}")
    return bad


def custom_v2() -> Model:
    return Model(a=3.1e-9, p=1.17, d0=55, speedup=3.3, gpu_overhead=0.7, gate=2.2e-8, gate_queries=1234,
                 save_bandwidth=4.4e8, bandwidth=1.1e8, bandwidth_out=3.3e7, shard_uses_store=True, mem_factor=2.5,
                 worker_fixed=0.83, worker_a=2.2e-8, worker_p=0.95, worker_d0=77, worker_gate=3.3e-8)


def test_xc_v2_calibrated_l4_is_the_block():
    l4, cal = Model.calibrated_l4(), Model.calibrated()
    print(f"Model.calibrated_l4() = {l4}")
    print("spec (decision 14, brief): calibrated() with worker_fixed 1.154, worker_a 1.295e-8, worker_p 0.891, worker_d0 154, "
          "worker_gate 2.07e-8, bandwidth 1.39e8, bandwidth_out 1.01e8, save_bandwidth 2.9e8; everything else as calibrated()")
    diff = {}
    for k in ALL_FIELDS:
        want = L4_FIELDS.get(k, getattr(cal, k))
        if getattr(l4, k) != pytest.approx(want, rel=1e-12) if not isinstance(want, bool) else getattr(l4, k) is not want:
            diff[k] = (getattr(l4, k), want)
    print(f"=> fields differing from the block: {diff}")
    assert not diff
    assert l4.gate_queries == 10000 and l4.shard_uses_store is True


def test_xc_v2_l4_totals_on_the_live_straggler():
    m = Model.calibrated_l4()
    g, h = got_v2(m, 408164, 128), hand_v2(m, 408164, 128)
    print("408164 x 128 on calibrated_l4(): " + ", ".join(f"{k} {v:.3f}" for k, v in g.items() if k not in ("vec", "idx", "mem")))
    print(f"by hand: worker build {h['worker_build']:.6f}, gpu total {h['gpu_total']:.6f}, local total {h['local']:.6f}")
    print("spec (brief): GPU total 13.109 s, local total 71.258 s")
    assert g["gpu_total"] == pytest.approx(13.109, abs=5e-4)
    assert g["local"] == pytest.approx(71.258, abs=5e-4)


def test_xc_v2_every_term_by_hand():
    bad = mismatches(Model.calibrated_l4(), "calibrated_l4()") + mismatches(custom_v2(), "custom v2")
    print(f"custom v2 model: {custom_v2()}")
    print(f"=> {len(SIZES)} sizes x {len(DIMS)} dims x 13 terms on calibrated_l4() and a custom v2 model: {len(bad)} mismatches")
    assert not bad, bad[:10]


def test_xc_v2_worker_a_zero_keeps_v1():
    m = Model(**{**{k: getattr(custom_v2(), k) for k in ALL_FIELDS}, "worker_a": 0.0, "worker_fixed": 7.0})
    print(f"worker_a 0 with a 7 s floor and the other worker terms set: {m}")
    print("spec (decision 14): when worker_a is 0 the v1 speedup form stands; worker_build_s is 0")
    bad = []
    for n in SIZES:
        for d in DIMS:
            g, h = v1x.got(m, n, d), v1x.hand(m, n, d)
            for k in g:
                tol = 1.0 if k in ("vec", "idx", "mem") else 1e-12 * abs(h[k]) + 1e-12
                if abs(g[k] - h[k]) > tol:
                    bad.append(f"n={n} dim={d} {k}: got {g[k]!r}, v1 {h[k]!r}")
            if m.worker_build_s(n, d) != 0:
                bad.append(f"n={n} dim={d}: worker_build_s {m.worker_build_s(n, d)} != 0")
    print(f"gpu_build_s(408164,128) {m.gpu_build_s(408164, 128):.6f} (v1 form {v1x.hand(m, 408164, 128)['gpu_build']:.6f}); "
          f"=> {len(bad)} mismatches")
    assert not bad, bad[:10]
    for ctor in (Model, Model.calibrated):
        mm = ctor()
        assert mm.worker_a == 0 and mm.worker_build_s(408164, 128) == 0, f"{ctor} must carry no worker shape"


def test_xc_v2_tiny_builds_pay_the_floor():
    print("spec (brief): a build of fewer than 2 vectors pays worker_fixed only (plus overhead and save in GPU build)")
    for m in (Model.calibrated_l4(), custom_v2()):
        for n in (0, 1):
            wb, gb = m.worker_build_s(n, 128), m.gpu_build_s(n, 128)
            want = m.worker_fixed + m.gpu_overhead + m.save_s(n, 128)
            print(f"  worker_fixed {m.worker_fixed} n={n}: worker_build {wb}, gpu_build {gb} (want {want}), cpu {m.cpu_build_s(n, 128)}")
            assert wb == pytest.approx(m.worker_fixed, abs=1e-12)
            assert gb == pytest.approx(want, abs=1e-12)


def test_xc_v2_from_env_selects_the_l4_block():
    m = Model.from_env({"COST_MODEL": "calibrated-l4"})
    print(f"from_env(COST_MODEL=calibrated-l4) = {m}")
    print("spec (brief): COST_MODEL=calibrated-l4, exactly that spelling, selects the GPU block on the worker as on the scheduler")
    assert m == Model.calibrated_l4()
    assert Model.from_env({"COST_MODEL": "calibrated"}) == Model.calibrated()
    assert Model.from_env({}) == Model()


@pytest.mark.parametrize("start", ["default", "calibrated", "calibrated-l4"])
@pytest.mark.parametrize("name", sorted(WORKER_NAMES))
def test_xc_v2_from_env_each_worker_name(start, name):
    attr, raw, val = WORKER_NAMES[name]
    base = {"default": Model(), "calibrated": Model.calibrated(), "calibrated-l4": Model.calibrated_l4()}[start]
    m = Model.from_env({"COST_MODEL": start, name: raw})
    print(f"COST_MODEL={start} {name}={raw}: {attr} {getattr(base, attr)!r} -> {getattr(m, attr)!r}")
    print("spec (brief): from_env reads COST_WORKER_FIXED/A/P/D0/GATE, the names the scheduler reads, on top of the selected model")
    assert getattr(m, attr) == pytest.approx(val)
    moved = [k for k in ALL_FIELDS if k != attr and getattr(m, k) != getattr(base, k)]
    assert not moved, f"{name} must change only {attr}; moved {moved}"


@pytest.mark.parametrize("spelling", ["Calibrated-L4", "calibrated_l4", "CALIBRATED-L4", "l4", "calibrated-l4 ", "calibratedl4"])
def test_xc_v2_from_env_refuses_other_spellings(spelling):
    print(f"spec (brief): exactly 'calibrated-l4'; the scheduler exits 2 on {spelling!r}, the worker mirrors it")
    with pytest.raises(Exception) as ei:
        Model.from_env({"COST_MODEL": spelling})
    print(f"  raises {type(ei.value).__name__}: {ei.value}")


def _start_worker(env_extra: dict[str, str], wait_s: float = 5.0) -> tuple[int | None, str]:
    """Start the worker process against an unreachable database: a refused configuration exits (2) before it
    connects; an accepted one is still retrying the database after wait_s (returned as None)."""
    env = {**os.environ, "DATABASE_URL": "postgres://x:y@127.0.0.1:1/none", "LOG_FORMAT": "text", **env_extra}
    try:
        p = subprocess.run([sys.executable, "-m", "kgpu_worker"], env=env, capture_output=True, text=True,
                           timeout=wait_s, cwd=Path(__file__).resolve().parents[1])
        return p.returncode, (p.stderr + p.stdout).strip()
    except subprocess.TimeoutExpired as t:
        out = (t.stderr or b"").decode() + (t.stdout or b"").decode()
        return None, out.strip()


BAD = [(n, v) for n in sorted(WORKER_NAMES) for v in ("-1", "nan", "inf")]


@pytest.mark.parametrize("name,value", BAD)
def test_xc_v2_worker_refuses_bad_worker_values(name, value):
    code, out = _start_worker({"COST_MODEL": "calibrated-l4", name: value})
    last = out.splitlines()[-1] if out else ""
    try:
        m = Model.from_env({"COST_MODEL": "calibrated-l4", name: value})
        fe = f"from_env accepts it ({WORKER_NAMES[name][0]}={getattr(m, WORKER_NAMES[name][0])}, gpu_total(408164,128)={m.gpu_total_s(408164, 128)})"
    except Exception as ex:  # noqa: BLE001 - recording what happens is the point
        fe = f"from_env raises {type(ex).__name__}"
    print(f"COST_MODEL=calibrated-l4 {name}={value}: worker process -> "
          f"{'still running after 5 s (accepted)' if code is None else f'exit {code}'}; {last[:200]}; {fe}")
    print("spec (brief): negative or non-finite COST_WORKER_* values are refused (the scheduler: Validate(), exit 2); "
          "the worker mirrors the scheduler (step 1 finding E: one env block must not mean two models)")
    assert code is not None and code != 0, f"the worker started with {name}={value}"


def test_xc_v2_worker_accepts_the_l4_block_and_refuses_other_spellings():
    code, out = _start_worker({"COST_MODEL": "calibrated-l4"})
    print(f"COST_MODEL=calibrated-l4 -> {'accepted (still running)' if code is None else f'exit {code}'}: {out.splitlines()[:3]}")
    assert code is None, out[-500:]
    for spelling in ("Calibrated-L4", "calibrated_l4"):
        code, out = _start_worker({"COST_MODEL": spelling})
        print(f"COST_MODEL={spelling!r} -> exit {code}: {out.splitlines()[-1:] if out else ''}")
        assert code not in (None, 0)


def test_xc_v2_fit_block_as_env_is_the_l4_block():
    """The v2 block `fit --v1 --v2 --fixed --drop-warmup` prints, pasted as an environment, is CalibratedL4 to its digits."""
    env = {"COST_MODEL": "calibrated", "COST_A": "2.3e-09", "COST_P": "1.220", "COST_D0": "187", "COST_GATE": "1.787e-08",
           "COST_GATE_QUERIES": "10000", "COST_WORKER_FIXED": "1.154", "COST_WORKER_A": "1.295e-08", "COST_WORKER_P": "0.891",
           "COST_WORKER_D0": "154", "COST_WORKER_GATE": "2.068e-08", "COST_GPU_OVERHEAD": "1.50", "COST_BANDWIDTH": "1.388e+08",
           "COST_BANDWIDTH_OUT": "1.007e+08", "COST_SAVE_BANDWIDTH": "2.914e+08"}
    m, l4 = Model.from_env(env), Model.calibrated_l4()
    print("the fit's printed v2 block (brief) applied through from_env, starting from COST_MODEL=calibrated as the block says")
    for n, d in ((10000, 128), (408164, 128), (120000, 960)):
        print(f"  {n} x {d}: gpu_total {m.gpu_total_s(n, d):.3f} (block {l4.gpu_total_s(n, d):.3f}), "
              f"local {m.local_total_s(n, d):.3f} (block {l4.local_total_s(n, d):.3f})")
        assert m.gpu_total_s(n, d) == pytest.approx(l4.gpu_total_s(n, d), rel=0.01)
        assert m.local_total_s(n, d) == pytest.approx(l4.local_total_s(n, d), rel=0.01)
    print("note: the block says COST_MODEL=calibrated, not calibrated-l4; it reaches the same model only because it sets every "
          "field in which the two differ")
