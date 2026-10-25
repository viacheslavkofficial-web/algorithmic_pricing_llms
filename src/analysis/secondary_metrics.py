# src/analysis/secondary_metrics.py
"""
Secondary pre-registered metrics for the duopoly identity study.

1. Time to collusion: first round after which the 10-round moving average
   of Delta stays above 0.5 until round 150 ("never" otherwise). Delta_t
   uses the mean price of both firms in the 10-round window ending at t,
   with run-specific Nash/monopoly benchmarks from metadata.json.

2. Reward-punishment regression (Fish et al.):
       p_i,t = a_ir + gamma * p_i,t-1 + delta * p_-i,t-1 + e
   with firm-run fixed effects (a_ir), rounds 101-150, standard errors
   clustered by run, estimated separately per condition.

Outputs:
    results/time_to_collusion.csv       (one row per run + medians)
    results/reward_punishment_regression.csv

Usage:
    uv run python src/analysis/secondary_metrics.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "experiments_synthetic" / "experiments_runs" / "2_agents"
RESULTS_DIR = REPO_ROOT / "results"

START_ROUND, END_ROUND, WINDOW, THRESHOLD = 101, 150, 10, 0.5


def load_run(run_dir: Path):
    with open(run_dir / "metadata.json") as f:
        metadata = json.load(f)
    df = pd.read_parquet(run_dir / "environment_history.parquet")
    return metadata, df


def time_to_collusion(metadata, prices_df) -> int | None:
    env = metadata["environment"]["environment_params"]
    nash = float(env["nash_prices"][0])
    monopoly = float(env["monopoly_prices"][0])
    market = prices_df.groupby("round")["price"].mean().sort_index()
    ma = market.rolling(WINDOW).mean().dropna()
    delta_ma = (ma - nash) / (monopoly - nash)
    above = delta_ma > THRESHOLD
    # Smallest window-end round t such that the condition holds for all
    # t' in [t, END_ROUND].
    candidates = [t for t in above.index if above.loc[t:END_ROUND].all()]
    return int(min(candidates)) if candidates else None


def regression_panel(metadata, prices_df) -> pd.DataFrame:
    """One row per (run, firm, round) with own and rival lagged prices."""
    rows = []
    for t in range(START_ROUND, END_ROUND + 1):
        by_agent = {}
        for agent in prices_df["agent"].unique():
            sub = prices_df[prices_df["agent"] == agent].set_index("round")["price"]
            if t - 1 in sub.index and t in sub.index:
                by_agent[agent] = (sub[t], sub[t - 1])
        agents = sorted(by_agent)
        for i, agent in enumerate(agents):
            rival = agents[1 - i]
            rows.append(
                {
                    "firm": agent,
                    "round": t,
                    "price": by_agent[agent][0],
                    "own_lag": by_agent[agent][1],
                    "rival_lag": by_agent[rival][1],
                }
            )
    return pd.DataFrame(rows)


def main():
    records = []
    panels = {}
    for run_dir in sorted(RUNS_DIR.glob("*duopoly_*")):
        metadata, prices_df = load_run(run_dir)
        condition = metadata["condition"]
        records.append(
            {
                "run_dir": run_dir.name,
                "condition": condition,
                "run_id": metadata["run_id"],
                "time_to_collusion": time_to_collusion(metadata, prices_df),
            }
        )
        panel = regression_panel(metadata, prices_df)
        panel["run_dir"] = run_dir.name
        panel["condition"] = condition
        panels.setdefault(condition, []).append(panel)

    ttc_df = pd.DataFrame(records)
    ttc_path = RESULTS_DIR / "time_to_collusion.csv"
    ttc_df.to_csv(ttc_path, index=False)

    medians = (
        ttc_df.groupby("condition")["time_to_collusion"]
        .median()
        .rename("median_time_to_collusion")
    )
    print("=== Time to collusion (10-round MA of Delta > 0.5 sustained to 150) ===")
    print(ttc_df.to_string(index=False))
    print()
    print(medians.to_string())
    print("(blank median = never in all runs of that condition)")

    reg_rows = []
    for condition, frames in sorted(panels.items()):
        data = pd.concat(frames, ignore_index=True)
        # Firm-run fixed effects via run x firm dummies; cluster by run.
        model = smf.ols(
            "price ~ own_lag + rival_lag + C(run_dir):C(firm)", data=data
        ).fit(cov_type="cluster", cov_kwds={"groups": data["run_dir"]})
        reg_rows.append(
            {
                "condition": condition,
                "gamma_own_lag": round(model.params["own_lag"], 4),
                "gamma_se": round(model.bse["own_lag"], 4),
                "delta_rival_lag": round(model.params["rival_lag"], 4),
                "delta_se": round(model.bse["rival_lag"], 4),
                "n_obs": int(model.nobs),
                "n_firm_runs": data.groupby(["run_dir", "firm"]).ngroups,
                "n_runs": data["run_dir"].nunique(),
            }
        )
        print(
            f"\n{condition}: gamma={model.params['own_lag']:.4f} "
            f"(SE {model.bse['own_lag']:.4f}), "
            f"delta={model.params['rival_lag']:.4f} "
            f"(SE {model.bse['rival_lag']:.4f}), N={int(model.nobs)}"
        )

    reg_df = pd.DataFrame(reg_rows)
    reg_path = RESULTS_DIR / "reward_punishment_regression.csv"
    reg_df.to_csv(reg_path, index=False)
    print(f"\nSaved: {ttc_path}")
    print(f"Saved: {reg_path}")


if __name__ == "__main__":
    main()
