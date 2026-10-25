# src/analysis/make_figures.py
"""
Report figures for the duopoly identity study.

Fig A: mean price per round by condition (average over runs and both
firms), shaded 95% CI across runs, with Nash and monopoly reference lines.
Fig B: run-level Delta by condition (one dot per run) with condition mean
and 95% CI.

Style: clean academic, readable in grayscale (distinct linestyles and
markers in addition to a consistent condition color map).

Outputs (PNG 300 dpi + PDF):
    results/figures/fig_a_price_trajectories.png/.pdf
    results/figures/fig_b_delta_by_condition.png/.pdf

Usage:
    uv run python src/analysis/make_figures.py
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
FIG_DIR = REPO_ROOT / "results" / "figures"

CONDITIONS = ["C0", "C1", "C2", "C3"]
COLORS = {"C0": "#555555", "C1": "#1f77b4", "C2": "#d62728", "C3": "#2ca02c"}
LINESTYLES = {"C0": "--", "C1": "-", "C2": "-.", "C3": ":"}
LABELS = {
    "C0": "C0 (no info)",
    "C1": "C1 (same model)",
    "C2": "C2 (diff. developer)",
    "C3": "C3 (diff. dev., matched)",
}
REFERENCE_NASH = 1.473
REFERENCE_MONOPOLY = 1.925


def load_prices_by_run():
    per_run = []
    for run_dir in sorted(RUNS_DIR.glob("*duopoly_*")):
        with open(run_dir / "metadata.json") as f:
            metadata = json.load(f)
        df = pd.read_parquet(run_dir / "environment_history.parquet")
        market = df.groupby("round")["price"].mean()
        per_run.append(
            {"condition": metadata["condition"], "run_dir": run_dir.name, "prices": market}
        )
    return per_run


def fig_a(per_run):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rounds = np.arange(1, 151)
    for condition in CONDITIONS:
        runs = [r for r in per_run if r["condition"] == condition]
        matrix = np.full((len(runs), 150), np.nan)
        for i, run in enumerate(runs):
            matrix[i] = run["prices"].reindex(rounds).to_numpy()
        mean = np.nanmean(matrix, axis=0)
        sem = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(matrix.shape[0])
        ax.plot(
            rounds,
            mean,
            color=COLORS[condition],
            linestyle=LINESTYLES[condition],
            linewidth=1.4,
            label=LABELS[condition],
        )
        ax.fill_between(
            rounds,
            mean - 1.96 * sem,
            mean + 1.96 * sem,
            color=COLORS[condition],
            alpha=0.15,
            linewidth=0,
        )
    ax.axhline(REFERENCE_NASH, color="black", linewidth=0.8, linestyle="-")
    ax.axhline(REFERENCE_MONOPOLY, color="black", linewidth=0.8, linestyle=":")
    ax.text(148, REFERENCE_NASH - 0.025, "Nash", ha="right", va="top", fontsize=8)
    ax.text(
        148, REFERENCE_MONOPOLY + 0.025, "Monopoly", ha="right", va="bottom", fontsize=8
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean price (both firms)")
    ax.set_xlim(1, 150)
    ax.set_ylim(1.35, 3.75)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig_a_price_trajectories.{ext}", dpi=300)
    plt.close(fig)


def fig_b(runs_df):
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    rng = np.random.default_rng(0)
    for i, condition in enumerate(CONDITIONS):
        deltas = runs_df[runs_df["condition"] == condition]["delta"].to_numpy()
        jitter = rng.uniform(-0.12, 0.12, size=len(deltas))
        ax.scatter(
            np.full(len(deltas), i) + jitter,
            deltas,
            s=18,
            color=COLORS[condition],
            alpha=0.7,
            edgecolors="none",
        )
        mean = deltas.mean()
        # 95% CI across runs using the t distribution (n = 10 per condition)
        ci = stats.t.ppf(0.975, len(deltas) - 1) * deltas.std(ddof=1) / np.sqrt(
            len(deltas)
        )
        ax.errorbar(
            i,
            mean,
            yerr=ci,
            fmt="D",
            color="black",
            markersize=5,
            capsize=4,
            linewidth=1.2,
        )
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS], fontsize=8)
    ax.set_ylabel(r"Collusion index $\Delta$ (rounds 101-150)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1, color="black", linewidth=0.8, linestyle=":")
    ax.text(3.45, 0.03, "Nash", ha="right", va="bottom", fontsize=8)
    ax.text(3.45, 1.03, "Monopoly", ha="right", va="bottom", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig_b_delta_by_condition.{ext}", dpi=300)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    per_run = load_prices_by_run()
    fig_a(per_run)
    runs_df = pd.read_csv(REPO_ROOT / "results" / "collusion_runs.csv")
    fig_b(runs_df)
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
