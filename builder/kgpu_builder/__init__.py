"""kgpu_builder: Phase 4's data tools and (from step 2) the hnswlib builder.

Everything here is run as a command line (``python -m kgpu_builder ...``)
so the Python worker and the Go shard execute the same code (Phase 4
decision 3). Configuration is by environment variable; see config.py.
"""
