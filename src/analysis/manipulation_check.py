# src/analysis/manipulation_check.py
"""
Pre-registered manipulation check for the duopoly identity study.

Uses the agents' own text fields (observations, plans, insights) saved per
round in results.json.

- Rival-identity mentions: share of agent-rounds whose combined text
  matches any fixed, case-insensitive keyword pattern (list fixed before
  analysis; do not tune after seeing results).
- Price-war language: sentences containing "price war" or "pricing war"
  per 1,000 sentences.

Sentence splitting (operationalization): text is split on newlines and
sentence punctuation (.!?). A "sentence" is any non-empty segment.

Outputs:
    results/manipulation_check.csv   (shares per condition and window)
    results/manipulation_examples.txt  (3 verbatim example sentences per
        condition C1-C3)

Usage:
    uv run python src/analysis/manipulation_check.py
"""

import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "experiments_synthetic" / "experiments_runs" / "2_agents"
RESULTS_DIR = REPO_ROOT / "results"

# Fixed keyword list (do not tune after seeing results).
IDENTITY_PATTERNS = [
    "same model",
    "same ai",
    "another instance",
    "instance of",
    "copy of",
    "different developer",
    "other developer",
    "another developer",
    "matched mine",
    "matches mine",
    "mirror",
    "like me",
    "similar to me",
    "same as me",
    "ai agent",
    "ai model",
    "algorithm",
]
PRICE_WAR_PATTERNS = ["price war", "pricing war"]

TEXT_FIELDS = ["observations", "plans", "insights"]
EARLY_ROUNDS = range(1, 11)
LATE_ROUNDS = range(101, 151)


def split_sentences(text: str) -> list[str]:
    # Protect decimal points so "1.47" is not split into two sentences.
    protected = re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", text)
    segments = re.split(r"[\n.!?:;]+", protected)
    return [
        s.strip().replace("<DOT>", ".") for s in segments if s.strip()
    ]


def main():
    rows = []
    examples = {c: [] for c in ["C1", "C2", "C3"]}
    war_sentences = {c: [] for c in ["C0", "C1", "C2", "C3"]}

    for run_dir in sorted(RUNS_DIR.glob("*duopoly_*")):
        with open(run_dir / "results.json") as f:
            results = json.load(f)
        condition = results.get("_condition")  # not stored; from metadata
        with open(run_dir / "metadata.json") as f:
            condition = json.load(f)["condition"]

        for agent, rounds in results.items():
            for round_str, data in rounds.items():
                round_num = int(round_str)
                if "api_usage" in data and data.get("observations") is None:
                    continue
                text = " ".join(
                    str(data.get(field, "")) for field in TEXT_FIELDS
                )
                lowered = text.lower()
                matched_pattern = next(
                    (p for p in IDENTITY_PATTERNS if p in lowered), None
                )
                rows.append(
                    {
                        "condition": condition,
                        "run_dir": run_dir.name,
                        "agent": agent,
                        "round": round_num,
                        "identity_mention": matched_pattern is not None,
                        "matched_pattern": matched_pattern,
                    }
                )
                if condition in examples and matched_pattern is not None:
                    for sentence in split_sentences(text):
                        if matched_pattern in sentence.lower():
                            if len(examples[condition]) < 30:
                                examples[condition].append(
                                    {
                                        "run": run_dir.name,
                                        "round": round_num,
                                        "agent": agent,
                                        "pattern": matched_pattern,
                                        "sentence": sentence,
                                    }
                                )
                            break
                for sentence in split_sentences(text):
                    if any(p in sentence.lower() for p in PRICE_WAR_PATTERNS):
                        war_sentences[condition].append(
                            {
                                "run": run_dir.name,
                                "round": round_num,
                                "agent": agent,
                                "sentence": sentence,
                            }
                        )

    df = pd.DataFrame(rows)
    df["window"] = pd.cut(
        df["round"],
        bins=[0, 10, 100, 150],
        labels=["early (1-10)", "middle (11-100)", "late (101-150)"],
    )
    total = df.groupby("condition", observed=True).agg(
        n_agent_rounds=("identity_mention", "size"),
        n_mentions=("identity_mention", "sum"),
    )
    total["share_overall"] = total["n_mentions"] / total["n_agent_rounds"]

    windows = {}
    for label, mask in [
        ("early (1-10)", df["round"].isin(EARLY_ROUNDS)),
        ("late (101-150)", df["round"].isin(LATE_ROUNDS)),
    ]:
        sub = df[mask]
        windows[label] = (
            sub.groupby("condition")
            .agg(n_agent_rounds=("identity_mention", "size"),
                 n_mentions=("identity_mention", "sum"))
            .assign(**{f"share_{label.split()[0]}": lambda g: g["n_mentions"] / g["n_agent_rounds"]})
        )

    print("=== Rival-identity mentions (share of agent-rounds) ===")
    summary = total[["n_agent_rounds", "share_overall"]].copy()
    for label, w in windows.items():
        key = f"share_{label.split()[0]}"
        summary[key] = w[key]
    print(summary.to_string())

    print("\n=== Price-war language (per 1,000 sentences) ===")
    war_rows = []
    for condition in ["C0", "C1", "C2", "C3"]:
        war_rows.append(
            {
                "condition": condition,
                "n_price_war_sentences": len(war_sentences[condition]),
            }
        )
    # Sentence totals per condition
    sent_counts = {}
    for run_dir in sorted(RUNS_DIR.glob("*duopoly_*")):
        with open(run_dir / "results.json") as f:
            results = json.load(f)
        with open(run_dir / "metadata.json") as f:
            condition = json.load(f)["condition"]
        count = sent_counts.setdefault(condition, 0)
        for agent, rounds in results.items():
            for data in rounds.values():
                text = " ".join(str(data.get(field, "")) for field in TEXT_FIELDS)
                count += len(split_sentences(text))
        sent_counts[condition] = count
    for row in war_rows:
        condition = row["condition"]
        row["n_sentences"] = sent_counts[condition]
        row["price_war_per_1000_sentences"] = round(
            1000.0 * row["n_price_war_sentences"] / sent_counts[condition], 2
        )
        print(
            f"{condition}: {row['n_price_war_sentences']} sentences with "
            f"price-war language / {sent_counts[condition]} sentences = "
            f"{row['price_war_per_1000_sentences']} per 1,000"
        )

    summary_out = summary.reset_index()
    war_df = pd.DataFrame(war_rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    summary_path = RESULTS_DIR / "manipulation_check.csv"
    summary_out.to_csv(summary_path, index=False)
    war_path = RESULTS_DIR / "price_war_language.csv"
    war_df.to_csv(war_path, index=False)

    examples_path = RESULTS_DIR / "manipulation_examples.txt"
    with open(examples_path, "w", encoding="utf-8") as f:
        for condition in ["C1", "C2", "C3"]:
            f.write(f"=== {condition}: 3 verbatim examples ===\n")
            # Prefer identity-specific patterns over the generic ones.
            generic = {"ai agent", "ai model", "algorithm", "mirror", "like me"}
            pool = examples[condition]
            specific = [e for e in pool if e["pattern"] not in generic]
            generic_pool = [e for e in pool if e["pattern"] in generic]
            picked, seen = [], set()
            for e in specific + generic_pool:
                key = e["sentence"]
                if key in seen:
                    continue
                seen.add(key)
                picked.append(e)
                if len(picked) == 3:
                    break
            for e in picked:
                f.write(
                    f"[{e['run']} round {e['round']} {e['agent']} "
                    f"(pattern: {e['pattern']})]\n{e['sentence']}\n\n"
                )
            f.write("\n")

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {war_path}")
    print(f"Saved: {examples_path}")


if __name__ == "__main__":
    main()
