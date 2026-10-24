# experiments_synthetic/check_balance.py
"""
Print remaining OpenRouter credit for the configured key (no key material
printed). Used before launching experiment batches; exit code 1 if the
remaining credit is below --min-usd.

Usage:
    uv run python experiments_synthetic/check_balance.py [--min-usd 1.0]
"""

import argparse
import json
import os
import sys
import urllib.request

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description="Check OpenRouter credit balance")
    parser.add_argument("--min-usd", type=float, default=1.0)
    args = parser.parse_args()

    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.load(resp)["data"]
    except Exception as e:
        print(f"Balance check failed: {e}")
        sys.exit(2)

    remaining = data["total_credits"] - data["total_usage"]
    print(
        f"total_credits={data['total_credits']:.2f} "
        f"total_usage={data['total_usage']:.4f} remaining={remaining:.4f}"
    )
    if remaining < args.min_usd:
        print(f"STOP: remaining credit below {args.min_usd:.2f} USD")
        sys.exit(1)


if __name__ == "__main__":
    main()
