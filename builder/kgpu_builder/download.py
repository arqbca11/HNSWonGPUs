"""Resumable download of a dataset file into the data directory.

A partial file is continued with an HTTP Range request, so an interrupted
3.8 GB download does not start over. The result is checked against the
catalogue's size; a size mismatch is an error, never silently truncated
data.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from .datasets import Dataset

CHUNK = 1 << 20


def download(ds: Dataset, data_dir: Path, *, log=print) -> Path:
    dest = data_dir / "ann-benchmarks" / ds.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if have == ds.size_bytes:
        log(f"{ds.name}: {dest} already complete ({have} bytes)")
        return dest
    if have > ds.size_bytes:
        raise SystemExit(f"{dest} is {have} bytes, larger than the catalogue's {ds.size_bytes}: delete it and retry")
    req = urllib.request.Request(ds.url)
    if have:
        req.add_header("Range", f"bytes={have}-")
        log(f"{ds.name}: resuming at {have} bytes")
    else:
        log(f"{ds.name}: downloading {ds.url} ({ds.size_bytes / 1e6:.0f} MB)")
    with urllib.request.urlopen(req) as resp, open(dest, "ab" if have else "wb") as out:
        if have and resp.status != 206:
            raise SystemExit(f"server did not honour the Range request (status {resp.status}); delete {dest} and retry")
        done = have
        while True:
            buf = resp.read(CHUNK)
            if not buf:
                break
            out.write(buf)
            done += len(buf)
            if done % (64 * CHUNK) < CHUNK:
                log(f"  {done / 1e6:.0f} / {ds.size_bytes / 1e6:.0f} MB", file=sys.stderr) if log is print else log(f"  {done / 1e6:.0f} / {ds.size_bytes / 1e6:.0f} MB")
    got = dest.stat().st_size
    if got != ds.size_bytes:
        raise SystemExit(f"{dest}: {got} bytes after download, catalogue says {ds.size_bytes}")
    log(f"{ds.name}: complete, {got} bytes")
    return dest
