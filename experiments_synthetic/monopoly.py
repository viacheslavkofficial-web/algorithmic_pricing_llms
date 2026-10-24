# experiments_synthetic/monopoly.py
# Monopoly competence-check pilot (as in Fish et al.), adapted for the
# AI Collusion Research Sprint (Oct 23-25, 2026) to be configurable via
# CLI arguments / env vars. Sprint defaults: 1 run, alpha = 1, 60 rounds,
# memory length 30, per-round plotting off.
import os
import sys
import asyncio
import argparse
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.LLM_agent import LLMAgent
from src.experiment.experiment import Experiment
from src.prompts.prompts import GENERAL_PROMPT, PP_P0
from src.prompts.prompts_models import create_pricing_response_model

from src.environment.calvano import CalvanoDemandEnvironment
from pathlib import Path

current_file_path = Path(__file__).resolve()

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "z-ai/glm-5.3-flash-20260826")


def parse_args():
    parser = argparse.ArgumentParser(description="Monopoly pricing pilot")
    parser.add_argument("--n-runs", type=int, default=int(os.getenv("N_RUNS", "1")))
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[float(a) for a in os.getenv("ALPHAS", "1").split(",")],
    )
    parser.add_argument(
        "--n-rounds", type=int, default=int(os.getenv("N_ROUNDS", "60"))
    )
    parser.add_argument(
        "--memory-length", type=int, default=int(os.getenv("MEMORY_LENGTH", "30"))
    )
    parser.add_argument(
        "--plotting",
        action="store_true",
        default=os.getenv("PLOTTING", "false").strip().lower() == "true",
        help="Enable per-round SVG plotting (disabled by default for speed)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=os.getenv("EXPERIMENT_NAME", "monopoly_pilot"),
    )
    return parser.parse_args()


async def main(alpha, args):
    PricingAgentResponse = create_pricing_response_model(
        include_wtp=True, wtp_value=4.51 * alpha
    )
    agents = [
        LLMAgent(
            "Firm A",
            prefix=PP_P0,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            response_model=PricingAgentResponse,
            memory_length=args.memory_length,
            prompt_template=GENERAL_PROMPT,
            env_params={"a": 2.0, "alpha": alpha, "c": 1.0},
        ),
    ]

    env = CalvanoDemandEnvironment(
        name="Calvano Market",
        description="Monopoly environment with Calvano 2020 demand",
    )

    experiment = Experiment(
        name=args.experiment_name,
        agents=agents,
        num_rounds=args.n_rounds,
        environment=env,
        experiment_dir=current_file_path.parent / "experiments_runs",
        experiment_plot=args.plotting,
        skip_plotting=not args.plotting,
    )
    await experiment.run()


if __name__ == "__main__":
    args = parse_args()
    for run in range(args.n_runs):
        for alpha in args.alphas:
            print(f"Running experiment {run + 1}/{args.n_runs} with alpha={alpha}")
            asyncio.run(main(alpha, args))
            print(f"Experiment with alpha={alpha} completed.\n")
