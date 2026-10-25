# src/analysis/profit_robustness.py
"""
Exploratory profit-based robustness analysis for the duopoly identity study.

The pre-registered price-based collusion index Delta stays primary and is
reported as-is (src/analysis/collusion_index.py). Everything here is
exploratory/robustness.

Profits used are REALISED profits from the run data
(environment_history.parquet, 'profit' column, rounds 101-150), averaged
over both firms per run - not profits implied by mean prices. The
theoretical benchmark profits pi_Nash and pi_Monopoly are taken from the
run's metadata.json (computed by the environment from its parameters).

Outputs:
    results/profit_robustness_runs.csv   (one row per run)
    results/profit_robustness_summary.csv (per-condition summary)
    results/profit_robustness_comparisons.csv (pairwise tests on Delta_pi)
    results/figures/fig_c_delta_vs_profit.png/.pdf

Usage:
    uv run python src/analysis/profit_robustness.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "experiments_synthetic" / "experiments_runs" / "2_agents"
RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

START_ROUND, END_ROUND = 101, 150
CONDITIONS = ["C0", "C1", "C2", "C3"]
COLORS = {"C0": "#555555", "C1": "#1f77b4", "C2": "#d62728", "C3": "#2ca02c"}


def load_run(run_dir: Path):
    with open(run_dir / "metadata.json") as f:
        metadata = json.load(f)
    df = pd.read_parquet(run_dir / "environment_history.parquet")
    return metadata, df


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = (
        ((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2)
        / (na + nb - 2)
    ) ** 0.5
    return (a.mean() - b.mean()) / pooled


def compare(df, c1, c2, column="delta_pi"):
    a = df[df.condition == c1][column]
    b = df[df.condition == c2][column]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    diff = a.mean() - b.mean()
    se = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 0.5
    u, up = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "comparison": f"{c1} vs {c2}",
        "mean_c1": round(a.mean(), 4),
        "mean_c2": round(b.mean(), 4),
        "diff": round(diff, 4),
        "ci95_lo": round(diff - 1.96 * se, 4),
        "ci95_hi": round(diff + 1.96 * se, 4),
        "welch_t": round(t, 4),
        "welch_p": round(p, 4),
        "cohens_d": round(cohens_d(a, b), 4),
        "mwu_u": round(u, 1),
        "mwu_p": round(up, 4),
    }


def main():
    rows = []
    for run_dir in sorted(RUNS_DIR.glob("*duopoly_*")):
        metadata, prices_df = load_run(run_dir)
        env = metadata["environment"]["environment_params"]
        pi_nash = float(np.mean(env["nash_profits"]))
        pi_monopoly = float(np.mean(env["monopoly_profits"]))

        window = prices_df[
            (prices_df["round"] >= START_ROUND)
            & (prices_df["round"] <= END_ROUND)
        ]
        per_firm_profit = window.groupby("agent")["profit"].mean()
        per_firm_quantity = window.groupby("agent")["quantity"].mean()
        pi_bar = float(per_firm_profit.mean())
        q_bar = float(per_firm_quantity.mean())
        delta_pi = (pi_bar - pi_nash) / (pi_monopoly - pi_nash)

        # Price dispersion within the run window
        price_sd = float(window["price"].std(ddof=1))
        rounds = sorted(window["round"].unique())
        gaps = []
        for t in rounds:
            p = window[window["round"] == t]["price"].to_numpy()
            if len(p) == 2:
                gaps.append(abs(p[0] - p[1]))
        mean_gap = float(np.mean(gaps))

        rows.append(
            {
                "run_dir": run_dir.name,
                "condition": metadata["condition"],
                "run_id": metadata["run_id"],
                "pi_bar_realised": round(pi_bar, 4),
                "q_bar_realised": round(q_bar, 4),
                "pi_nash": round(pi_nash, 4),
                "pi_monopoly": round(pi_monopoly, 4),
                "delta_pi": round(delta_pi, 4),
                "delta_price": round(
                    (float(window["price"].mean()) - float(env["nash_prices"][0]))
                    / (float(env["monopoly_prices"][0]) - float(env["nash_prices"][0])),
                    4,
                ),
                "price_sd_within_run": round(price_sd, 4),
                "mean_abs_price_gap": round(mean_gap, 4),
            }
        )

    df = pd.DataFrame(rows)
    runs_path = RESULTS_DIR / "profit_robustness_runs.csv"
    df.to_csv(runs_path, index=False)

    summary = (
        df.groupby("condition")
        .agg(
            n=("delta_pi", "size"),
            mean_delta_pi=("delta_pi", "mean"),
            sd_delta_pi=("delta_pi", lambda s: s.std(ddof=1)),
            median_delta_pi=("delta_pi", "median"),
            mean_pi_bar=("pi_bar_realised", "mean"),
            mean_q_bar=("q_bar_realised", "mean"),
            mean_price_sd=("price_sd_within_run", "mean"),
            mean_price_gap=("mean_abs_price_gap", "mean"),
        )
        .round(4)
    )
    summary_path = RESULTS_DIR / "profit_robustness_summary.csv"
    summary.to_csv(summary_path)

    comparisons = pd.DataFrame(
        [
            compare(df, "C1", "C2"),
            compare(df, "C3", "C2"),
            compare(df, "C3", "C1"),
            compare(df, "C1", "C0"),
            compare(df, "C2", "C0"),
        ]
    )
    comp_path = RESULTS_DIR / "profit_robustness_comparisons.csv"
    comparisons.to_csv(comp_path, index=False)

    print("Profits used: REALISED per-firm profits from environment_history.parquet")
    print("(rounds 101-150, averaged over both firms); benchmark profits from")
    print("metadata.json environment parameters.\n")
    print("=== Per-run (first rows) ===")
    print(df.drop(columns=["run_dir"]).to_string(index=False))
    print("\n=== Per-condition summary ===")
    print(summary.to_string())
    print("\n=== Pairwise comparisons on Delta_pi ===")
    print(comparisons.to_string(index=False))

    # Overshoot analysis (price-based Delta, the primary outcome, used only
    # for the overshoot definition)
    print("\n=== Overshoot (Delta_price > 1) ===")
    over = {}
    for condition in CONDITIONS:
        deltas = df[df.condition == condition]["delta_price"]
        over[condition] = int((deltas > 1).sum())
        print(
            f"{condition}: {over[condition]}/{len(deltas)} runs above monopoly "
            f"({over[condition] / len(deltas):.0%})"
        )
    fisher_table = [
        [over["C1"], 10 - over["C1"]],
        [over["C2"], 10 - over["C2"]],
    ]
    odds, fisher_p = stats.fisher_exact(fisher_table)
    print(f"Fisher exact C1 vs C2: table {fisher_table}, odds={odds:.4f}, p={fisher_p:.4f}")

    r = stats.pearsonr(df["delta_price"], df["pi_bar_realised"])
    rho = stats.spearmanr(df["delta_price"], df["pi_bar_realised"])
    print(
        f"Correlation Delta_price vs realised per-firm profit: Pearson r={r.statistic:.4f} "
        f"(p={r.pvalue:.2e}), Spearman rho={rho.statistic:.4f} (p={rho.pvalue:.2e})"
    )

    # Sensitivity: primary C1 vs C2 test excluding runs with Delta_price > 2.5
    excluded = df[df["delta_price"] > 2.5][["condition", "run_dir", "delta_price"]]
    n_excluded = {c: int((excluded.condition == c).sum()) for c in CONDITIONS}
    print("\n=== Sensitivity: C1 vs C2 excluding runs with Delta_price > 2.5 ===")
    print("Exclusion rule (stated): drop any run with price-based Delta > 2.5 "
          "in rounds 101-150, from either comparison group.")
    print(f"Excluded runs: {n_excluded}")
    sens = pd.DataFrame([compare(df[df["delta_price"] <= 2.5], "C1", "C2")])
    print(sens.to_string(index=False))

    # Figure C: Delta vs realised per-firm profit
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for condition in CONDITIONS:
        sub = df[df.condition == condition]
        ax.scatter(
            sub["delta_price"],
            sub["pi_bar_realised"],
            s=26,
            color=COLORS[condition],
            alpha=0.75,
            edgecolors="none",
            label=condition,
        )
    ax.axvline(1, color="black", linewidth=0.8, linestyle=":")
    pi_nash_mean = df["pi_nash"].mean()
    pi_mono_mean = df["pi_monopoly"].mean()
    ax.axhline(pi_nash_mean, color="black", linewidth=0.8)
    ax.axhline(pi_mono_mean, color="black", linewidth=0.8, linestyle=":")
    ax.text(ax.get_xlim()[1], pi_nash_mean - 0.4, "Nash profit", ha="right", va="top", fontsize=8)
    ax.text(ax.get_xlim()[1], pi_mono_mean + 0.4, "Monopoly profit", ha="right", va="bottom", fontsize=8)
    ax.set_xlabel(r"Price-based collusion index $\Delta$ (rounds 101-150)")
    ax.set_ylabel("Realised per-firm profit (mean, rounds 101-150)")
    ax.legend(frameon=False, fontsize=8, title=None)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig_c_delta_vs_profit.{ext}", dpi=300)
    plt.close(fig)

    print(f"\nSaved: {runs_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {comp_path}")
    print(f"Saved: {FIG_DIR / 'fig_c_delta_vs_profit.png'} and .pdf")


if __name__ == "__main__":
    main()
