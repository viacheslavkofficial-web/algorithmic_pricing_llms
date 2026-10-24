# src/analysis/collusion_index.py
"""
Collusion index analysis for duopoly identity runs (sprint study).

For each duopoly run folder this script reads metadata.json and
environment_history.parquet, takes the Nash and monopoly benchmark prices
from the run's metadata (verifying they match the theoretical duopoly
values of about 1.473 and 1.925), and computes the pre-registered primary
outcome:

    p_bar = mean price of both firms over rounds 101-150
    Delta = (p_bar - p_Nash) / (p_monopoly - p_Nash)

It also aggregates per-call api_usage (latency, tokens, cost) and failed
attempts from results.json.

Outputs one CSV row per run plus a per-condition summary, both printed
and saved to results/ (tracked; raw run folders stay untracked).

Usage:
    uv run python src/analysis/collusion_index.py
    uv run python src/analysis/collusion_index.py --runs-dir <dir> [dirs...]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = (
    REPO_ROOT / "experiments_synthetic" / "experiments_runs" / "2_agents"
)
RESULTS_DIR = REPO_ROOT / "results"

# Theoretical duopoly benchmarks at a=2, a_0=0, mu=0.25, c=1, beta=100,
# alpha=1 (SPRINT_BRIEF.md). The script warns when metadata deviates.
EXPECTED_NASH = 1.473
EXPECTED_MONOPOLY = 1.925
TOLERANCE = 0.01


def analyze_run(run_dir: Path, start_round: int, end_round: int) -> dict:
    with open(run_dir / "metadata.json") as f:
        metadata = json.load(f)
    prices_df = pd.read_parquet(run_dir / "environment_history.parquet")
    with open(run_dir / "results.json") as f:
        results = json.load(f)

    env_params = metadata["environment"]["environment_params"]
    nash = float(env_params["nash_prices"][0])
    monopoly = float(env_params["monopoly_prices"][0])
    if abs(nash - EXPECTED_NASH) > TOLERANCE or abs(monopoly - EXPECTED_MONOPOLY) > TOLERANCE:
        print(
            f"⚠️ {run_dir.name}: benchmarks off expected values "
            f"(nash {nash:.4f} vs ~{EXPECTED_NASH}, "
            f"monopoly {monopoly:.4f} vs ~{EXPECTED_MONOPOLY})"
        )

    window = prices_df[
        (prices_df["round"] >= start_round) & (prices_df["round"] <= end_round)
    ]
    p_bar = float(window["price"].mean())
    delta = (p_bar - nash) / (monopoly - nash)

    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_reasoning_tokens = 0
    calls = 0
    failed_attempts = 0
    parse_failures = 0
    for agent_rounds in results.values():
        for round_data in agent_rounds.values():
            usage = round_data.get("api_usage")
            if usage is None:
                continue
            calls += 1
            total_cost += usage.get("cost_usd") or 0.0
            total_prompt_tokens += usage.get("prompt_tokens") or 0
            total_completion_tokens += usage.get("completion_tokens") or 0
            total_reasoning_tokens += usage.get("reasoning_tokens") or 0
            failed_attempts += max((usage.get("attempt", 1) or 1) - 1, 0)
            parse_failures += usage.get("parse_failures", 0) or 0

    return {
        "run_dir": run_dir.name,
        "condition": metadata.get("condition"),
        "run_id": metadata.get("run_id"),
        "n_rounds": metadata.get("num_rounds"),
        "model": list(metadata.get("agents_models", {}).values())[0],
        "temperature": list(metadata.get("agents_temperature", {}).values())[0],
        "reasoning_effort": list(metadata.get("agents_reasoning_effort", {}).values())[0],
        "p_nash": round(nash, 4),
        "p_monopoly": round(monopoly, 4),
        "p_bar": round(p_bar, 4),
        "delta": round(delta, 4),
        "above_nash": p_bar > nash,
        "n_calls": calls,
        "total_cost_usd": round(total_cost, 6),
        "mean_prompt_tokens": round(total_prompt_tokens / calls, 1) if calls else None,
        "mean_completion_tokens": round(total_completion_tokens / calls, 1)
        if calls
        else None,
        "mean_reasoning_tokens": round(total_reasoning_tokens / calls, 1)
        if calls
        else None,
        "failed_attempts": failed_attempts,
        "parse_failures": parse_failures,
    }


def main():
    parser = argparse.ArgumentParser(description="Collusion index analysis")
    parser.add_argument(
        "run_dirs",
        type=str,
        nargs="*",
        help="Run folders to analyze (default: all duopoly_* folders under "
        "experiments_synthetic/experiments_runs/2_agents)",
    )
    parser.add_argument("--start-round", type=int, default=101)
    parser.add_argument("--end-round", type=int, default=150)
    args = parser.parse_args()

    if args.run_dirs:
        run_dirs = [Path(d) for d in args.run_dirs]
    else:
        run_dirs = sorted(
            d for d in DEFAULT_RUNS_DIR.glob("*duopoly_*") if d.is_dir()
        ) if DEFAULT_RUNS_DIR.exists() else []
    if not run_dirs:
        print("No run folders found.")
        sys.exit(1)

    rows = [
        analyze_run(run_dir, args.start_round, args.end_round)
        for run_dir in sorted(run_dirs)
    ]
    runs_df = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(exist_ok=True)
    runs_path = RESULTS_DIR / "collusion_runs.csv"
    runs_df.to_csv(runs_path, index=False)

    print(f"Found {len(runs_df)} runs")
    print(f"\nBenchmarks verified against expected nash ~{EXPECTED_NASH}, "
          f"monopoly ~{EXPECTED_MONOPOLY} (tolerance {TOLERANCE}); see warnings above.")
    print(f"\n=== Per-run results (rounds {args.start_round}-{args.end_round}) ===")
    print(runs_df.to_string(index=False))

    summary_rows = []
    for condition, group in runs_df.groupby("condition", dropna=False):
        summary_rows.append(
            {
                "condition": condition,
                "n": len(group),
                "mean_delta": round(group["delta"].mean(), 4),
                "sd_delta": round(group["delta"].std(ddof=1), 4)
                if len(group) > 1
                else None,
                "runs_above_nash": int(group["above_nash"].sum()),
                "mean_p_bar": round(group["p_bar"].mean(), 4),
                "total_cost_usd": round(group["total_cost_usd"].sum(), 6),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "collusion_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Per-condition summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {runs_path}")
    print(f"Saved: {summary_path}")

    failed = runs_df[runs_df["failed_attempts"] > 0]
    if len(failed):
        print(f"\n⚠️ Runs with failed attempts (check pre-registration exclusion "
              f"rule for >5% API failures): {', '.join(failed['run_dir'])}")


if __name__ == "__main__":
    main()
