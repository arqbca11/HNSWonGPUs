"""The probe server answers on its own thread, independent of the main loop."""

import json
import socket
import time
import urllib.request

from kgpu_worker.health import Health


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_livez_always_and_healthz_follows_heartbeats():
    h = Health(stale_after_s=0.3)
    port = free_port()
    srv = h.serve(f"127.0.0.1:{port}")
    try:
        base = f"http://127.0.0.1:{port}"
        code, body = get(base + "/livez")
        print(f"/livez before any heartbeat -> {code} {body}")
        assert code == 200
        code, body = get(base + "/healthz")
        print(f"/healthz before any heartbeat -> {code} {body}")
        assert code == 503
        h.heartbeat_ok()
        code, body = get(base + "/healthz")
        print(f"/healthz right after a heartbeat -> {code} {body}")
        assert code == 200
        time.sleep(0.45)
        code, body = get(base + "/healthz")
        print(f"/healthz 0.45 s later (stale_after 0.3 s) -> {code} {body}")
        assert code == 503
        code, _ = get(base + "/livez")
        assert code == 200
        print("liveness never depends on Postgres; readiness ages out with the heartbeat")
    finally:
        srv.shutdown()


def test_metrics_served_on_the_health_port():
    """Phase 2 step 5: /metrics on the probe port lists the worker's metrics, every
    builds_total series present from the start so dashboards never see a gap."""
    from kgpu_worker import metrics  # noqa: F401  registers the metrics

    h = Health(stale_after_s=5)
    port = free_port()
    srv = h.serve(f"127.0.0.1:{port}")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as r:
            assert r.status == 200
            assert r.headers["Content-Type"].startswith("text/plain")
            text = r.read().decode()
        for name in ("build_duration_seconds_bucket", "worker_busy", 'builds_total{outcome="done"}',
                     'builds_total{outcome="released"}', 'builds_total{outcome="lost"}', 'builds_total{outcome="failed"}'):
            assert name in text, name
        print("/metrics served on the probe port; worker_busy and every builds_total outcome present")
    finally:
        srv.shutdown()
