# experiments_synthetic/run_parallel.py
"""
Parallel launcher for duopoly_identity runs (sprint study).

Starts N experiment runs as separate OS processes (each run is sequential
in rounds but independent across runs), at most --max-parallel at a time,
and waits for all of them. Each process writes its own console log under
experiments_runs/launcher_logs/.

Usage:
    python experiments_synthetic/run_parallel.py --runs C0:1 C0:2 C0:3 C0:4 C0:5
    python experiments_synthetic/run_parallel.py --runs C1:1 C1:2 --max-parallel 2

Exit code is non-zero if any run failed; failed run pairs are printed so the
discarded-and-rerun rule of the pre-registration can be applied.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "duopoly_identity.py"
LOG_DIR = Path(__file__).resolve().parent / "experiments_runs" / "launcher_logs"


def parse_args():
    parser = argparse.ArgumentParser(description="Launch duopoly runs in parallel")
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        required=True,
        help="List of runs as CONDITION:RUN_ID, e.g. C0:1 C0:2",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=5,
        help="Maximum number of concurrent run processes (default 5)",
    )
    args = parser.parse_args()
    pairs = []
    for spec in args.runs:
        condition, _, run_id = spec.partition(":")
        if not condition or not run_id.isdigit():
            parser.error(f"Invalid run spec {spec!r}; expected CONDITION:RUN_ID")
        pairs.append((condition, int(run_id)))
    return pairs, args.max_parallel


def main():
    pairs, max_parallel = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    pending = list(pairs)
    running = []  # (condition, run_id, Popen, start_time)
    finished = []  # (condition, run_id, returncode, elapsed_seconds)

    def launch(condition, run_id):
        log_path = LOG_DIR / f"duopoly_{condition}_run{run_id}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        process = subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT),
                "--condition",
                condition,
                "--run-id",
                str(run_id),
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        print(
            f"[launcher] started {condition}:{run_id} (pid {process.pid}, "
            f"log {log_path})",
            flush=True,
        )
        return process

    while pending or running:
        while pending and len(running) < max_parallel:
            condition, run_id = pending.pop(0)
            process = launch(condition, run_id)
            running.append((condition, run_id, process, time.time()))

        still_running = []
        for condition, run_id, process, start in running:
            code = process.poll()
            if code is None:
                still_running.append((condition, run_id, process, start))
            else:
                elapsed = time.time() - start
                finished.append((condition, run_id, code, elapsed))
                status = "OK" if code == 0 else f"FAILED (exit {code})"
                print(
                    f"[launcher] finished {condition}:{run_id} in "
                    f"{elapsed / 60:.1f} min: {status}",
                    flush=True,
                )
        running = still_running
        time.sleep(5)

    failed = [(c, r) for c, r, code, _ in finished if code != 0]
    print(f"[launcher] all {len(finished)} runs finished; {len(failed)} failed")
    if failed:
        print("[launcher] failed runs (candidates for discard-and-rerun):")
        for condition, run_id in failed:
            print(f"  {condition}:{run_id}")
        sys.exit(1)


if __name__ == "__main__":
    main()
