import sys
import subprocess

PYTHON = sys.executable

jobs = [
    [
        PYTHON,
        "scripts/run_batch.py",
        "--config", "configs/env.yaml",
        "--lookahead-config", "configs/lookahead_baseline.yaml",
        "--n-episodes", "2000",
        "--seed-start", "0",
    ],
    [
        PYTHON,
        "scripts/run_batch.py",
        "--config", "configs/env.yaml",
        "--lookahead-config", "configs/lookahead_conservative.yaml",
        "--n-episodes", "2000",
        "--seed-start", "0",
    ],
    [
        PYTHON,
        "scripts/run_batch.py",
        "--config", "configs/env.yaml",
        "--lookahead-config", "configs/lookahead_optimistic.yaml",
        "--n-episodes", "2000",
        "--seed-start", "0",
    ],
    [
        PYTHON,
        "scripts/run_batch.py",
        "--config", "configs/env.yaml",
        "--agent", "myopic",
        "--n-episodes", "2000",
        "--seed-start", "0",
    ],
]

procs = [subprocess.Popen(job) for job in jobs]

for p in procs:
    p.wait()

print("All batches completed.")