# src/analysis/condition_tests.py
"""
Pre-registered primary and secondary comparisons on the price-based
collusion index Delta (rounds 101-150), from results/collusion_runs.csv.

- Primary: C1 vs C2. Secondary: C3 vs C2, C3 vs C1.
- Welch two-sided t-test; 95% CI uses the t distribution with
  Welch-Satterthwaite degrees of freedom; Cohen's d; Mann-Whitney U.

Output: results/condition_tests.csv

Usage:
    uv run python src/analysis/condition_tests.py
"""

from pathlib import Path

import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = (
        ((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2)
        / (na + nb - 2)
    ) ** 0.5
    return (a.mean() - b.mean()) / pooled


def welch_ci(a, b, alpha=0.05):
    diff = a.mean() - b.mean()
    va = a.var(ddof=1) / len(a)
    vb = b.var(ddof=1) / len(b)
    se = (va + vb) ** 0.5
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    crit = stats.t.ppf(1 - alpha / 2, df)
    return diff, df, diff - crit * se, diff + crit * se


def compare(df, c1, c2, column="delta"):
    a = df[df.condition == c1][column]
    b = df[df.condition == c2][column]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    diff, df_ws, ci_lo, ci_hi = welch_ci(a, b)
    u, up = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "comparison": f"{c1} vs {c2}",
        "outcome": column,
        "mean_c1": round(a.mean(), 4),
        "sd_c1": round(a.std(ddof=1), 4),
        "mean_c2": round(b.mean(), 4),
        "sd_c2": round(b.std(ddof=1), 4),
        "diff": round(diff, 4),
        "ci95_lo": round(ci_lo, 4),
        "ci95_hi": round(ci_hi, 4),
        "welch_t": round(t, 4),
        "welch_df": round(df_ws, 2),
        "welch_p": round(p, 4),
        "cohens_d": round(cohens_d(a, b), 4),
        "mwu_u": round(u, 1),
        "mwu_p": round(up, 4),
    }


def main():
    runs = pd.read_csv(RESULTS_DIR / "collusion_runs.csv")
    tests = pd.DataFrame(
        [
            compare(runs, "C1", "C2"),
            compare(runs, "C3", "C2"),
            compare(runs, "C3", "C1"),
        ]
    )
    out_path = RESULTS_DIR / "condition_tests.csv"
    tests.to_csv(out_path, index=False)
    print(tests.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
