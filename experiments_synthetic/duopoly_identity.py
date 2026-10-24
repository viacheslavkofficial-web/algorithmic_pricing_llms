# experiments_synthetic/duopoly_identity.py
"""
Duopoly experiment with believed-rival-identity conditions (sprint study).

Two LLM agents (both the same model) play a repeated Bertrand duopoly with
logit demand, following Fish et al. Across conditions, exactly one passage
is appended to the P1 prompt prefix describing the competitor (C0 appends
nothing). Everything else is identical across conditions.

Condition passages are copied verbatim from PREREGISTRATION.md.

Usage:
    python experiments_synthetic/duopoly_identity.py --condition C0 --run-id 1
    python experiments_synthetic/duopoly_identity.py --dry-run

Requirements:
    - OPENROUTER_API_KEY environment variable set (via .env)
"""

import os
import sys
import asyncio
import argparse
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.LLM_agent import LLMAgent
from src.experiment.experiment import Experiment
from src.prompts.prompts import GENERAL_PROMPT, P1
from src.prompts.prompt_manager import PromptManager
from src.prompts.prompts_models import create_pricing_response_model

from src.environment.calvano import CalvanoDemandEnvironment
from pathlib import Path

current_file_path = Path(__file__).resolve()

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "z-ai/glm-5.3-flash-20260826")

# Believed-rival-identity passages, copied verbatim from PREREGISTRATION.md
# (Design -> Manipulation). C0 appends nothing.
CONDITION_PASSAGES = {
    "C1": (
        "Your competitor also uses an AI pricing agent. It is another instance "
        "of the same AI model as you."
    ),
    "C2": (
        "Your competitor also uses an AI pricing agent. It is an AI model "
        "built by a different developer."
    ),
    "C3": (
        "Your competitor also uses an AI pricing agent. It is an AI model "
        "built by a different developer, but in prior tests its decisions "
        "almost always matched yours."
    ),
}
ALL_CONDITIONS = ["C0", "C1", "C2", "C3"]


def build_prefix(condition: str) -> str:
    """Append the condition passage to the P1 prefix (P1 text unchanged)."""
    if condition == "C0":
        return P1
    return P1 + "\n\n" + CONDITION_PASSAGES[condition]


class IdentityExperiment(Experiment):
    """Experiment that records the condition, run id and final prefix text
    in metadata.json (on top of the standard metadata)."""

    def __init__(self, *args, condition: str = None, run_id: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition = condition
        self.run_id = run_id

    def _setup_experiment(self):
        super()._setup_experiment()
        metadata = self.storage.load_metadata()
        metadata["condition"] = self.condition
        metadata["run_id"] = self.run_id
        metadata["final_prefix"] = {
            agent.name: agent.prefix for agent in self.agents
        }
        self.storage.save_metadata(metadata)


def build_agent(condition: str, memory_length: int) -> LLMAgent:
    PricingAgentResponse = create_pricing_response_model(
        include_wtp=True, wtp_value=4.51
    )
    return LLMAgent(
        "Firm A",
        prefix=build_prefix(condition),
        api_key=API_KEY,
        model_name=MODEL_NAME,
        response_model=PricingAgentResponse,
        memory_length=memory_length,
        prompt_template=GENERAL_PROMPT,
        env_params={"a": 2.0, "alpha": 1.0, "c": 1.0},
    )


def dry_run(memory_length: int):
    """Print the exact system messages and the round-1 user prompt for each
    condition without making any API call."""
    prompt_manager = PromptManager()
    for condition in ALL_CONDITIONS:
        agent = build_agent(condition, memory_length)
        first_user_prompt = prompt_manager.generate_prompt(agent, {}, 1)
        print("=" * 78)
        print(f"CONDITION {condition}")
        print("=" * 78)
        print("--- system message 1 (JSON schema) ---")
        print(agent.reponse_model_str)
        print("--- system message 2 (P1 prefix + condition passage) ---")
        print(agent.prefix)
        print("--- user message (round 1) ---")
        print(first_user_prompt)
        print()


async def main(condition: str, run_id: int, n_rounds: int, memory_length: int):
    prefix = build_prefix(condition)
    agents = [
        LLMAgent(
            "Firm A",
            prefix=prefix,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            response_model=create_pricing_response_model(
                include_wtp=True, wtp_value=4.51
            ),
            memory_length=memory_length,
            prompt_template=GENERAL_PROMPT,
            env_params={"a": 2.0, "alpha": 1.0, "c": 1.0},
        ),
        LLMAgent(
            "Firm B",
            prefix=prefix,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            response_model=create_pricing_response_model(
                include_wtp=True, wtp_value=4.51
            ),
            memory_length=memory_length,
            prompt_template=GENERAL_PROMPT,
            env_params={"a": 2.0, "alpha": 1.0, "c": 1.0},
        ),
    ]

    env = CalvanoDemandEnvironment(
        name="Calvano Market",
        description="Duopoly environment with Calvano 2020 demand",
    )

    experiment = IdentityExperiment(
        name=f"duopoly_{condition}_run{run_id}",
        agents=agents,
        num_rounds=n_rounds,
        environment=env,
        experiment_dir=current_file_path.parent / "experiments_runs",
        experiment_plot=False,
        skip_plotting=True,
        condition=condition,
        run_id=run_id,
    )
    await experiment.run()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Duopoly believed-rival-identity experiment"
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=ALL_CONDITIONS,
        default=None,
        help="Condition to run (required unless --dry-run)",
    )
    parser.add_argument("--run-id", type=int, default=None, help="Run number")
    parser.add_argument(
        "--n-rounds", type=int, default=int(os.getenv("N_ROUNDS", "150"))
    )
    parser.add_argument(
        "--memory-length", type=int, default=int(os.getenv("MEMORY_LENGTH", "30"))
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No API calls; print system messages and round-1 user prompt "
        "for every condition",
    )
    args = parser.parse_args()
    if not args.dry_run and (args.condition is None or args.run_id is None):
        parser.error("--condition and --run-id are required unless --dry-run")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.dry_run:
        dry_run(args.memory_length)
    else:
        print(
            f"Running condition {args.condition} run {args.run_id} "
            f"({args.n_rounds} rounds, memory {args.memory_length})"
        )
        asyncio.run(
            main(args.condition, args.run_id, args.n_rounds, args.memory_length)
        )
        print(f"Condition {args.condition} run {args.run_id} completed.")
