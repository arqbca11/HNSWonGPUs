#!/usr/bin/env bash
# The shard VM's first boot (deploy/gcp): the toolchain and the repo, so the
# shard simulator and the builder run here as the database fleet's host.
# Re-run safe. Logs to /var/log/kgpu-startup.log; `kgpu-ready` is touched at
# the end.
exec >> /var/log/kgpu-startup.log 2>&1
set -eux
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get install -y git curl build-essential python3 python3-venv ca-certificates
# Go (for the shard simulator)
if [ ! -x /usr/local/go/bin/go ]; then
  curl -fsSL https://go.dev/dl/go1.27.1.linux-amd64.tar.gz -o /tmp/go.tgz && rm -rf /usr/local/go && tar -C /usr/local -xzf /tmp/go.tgz
fi
# uv (for the builder's Python)
if [ ! -x /usr/local/bin/uv ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
fi
mkdir -p /opt/kgpu && cd /opt/kgpu
if [ ! -d HNSWonGPUs ]; then git clone https://github.com/arqbca11/HNSWonGPUs.git; fi
cd HNSWonGPUs && git pull --ff-only || true
export PATH="/usr/local/go/bin:/usr/local/bin:$PATH"
(cd builder && uv python install 3.12 && uv sync --frozen)
go build ./... || true
chmod -R a+rwX /opt/kgpu
touch /opt/kgpu/kgpu-ready
