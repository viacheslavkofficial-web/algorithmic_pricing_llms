"""
LLM Agent Implementation for Market Simulation

This module provides the LLMAgent class that interfaces with an
OpenAI-compatible API (OpenRouter) for strategic pricing decisions in
oligopoly market experiments.
"""

import json
import logging
import os
import asyncio
import re
import time

from openai import AsyncOpenAI, APIStatusError
from typing import Dict, Type
from pydantic import BaseModel, ValidationError, create_model

from src.agents.base_agent import Agent

MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 1

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL_NAME = "z-ai/glm-5.3-flash-20260826"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_REASONING_EFFORT = "low"


def _extract_first_json_object(text: str) -> dict:
    """
    Extract the first JSON object from an LLM response string.

    Tries, in order: the raw string, the content of the first markdown
    code fence, and the first balanced {...} block anywhere in the text.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    candidates = [text.strip()]

    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())

    start = text.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError("No valid JSON object found in model response")


class LLMAgent(Agent):
    """
    LLM-based agent for strategic pricing decisions in oligopoly markets.

    This agent calls an OpenAI-compatible chat completions API (OpenRouter)
    to make pricing decisions based on market history and strategic context.
    It maintains a rolling memory window and validates responses using
    Pydantic models.

    Args:
        name: Unique identifier for the agent
        prefix: Prompt prefix defining agent's strategic context
        api_key: OpenRouter API key (falls back to OPENROUTER_API_KEY env var)
        model_name: OpenRouter model ID (falls back to MODEL_NAME env var,
            default 'z-ai/glm-5.3-flash-20260826')
        response_model: Pydantic model for response validation
        prompt_template: Optional template for prompt formatting
        memory_length: Number of periods to maintain in memory (default: 100)
        env_params: Market environment parameters
        logger: Logger instance for experiment tracking
        temperature: Sampling temperature (falls back to TEMPERATURE env var,
            default 0.7, the value used by Fish et al.)
        reasoning_effort: Reasoning effort passed to OpenRouter via
            extra_body={"reasoning": {"effort": ...}} (falls back to
            REASONING_EFFORT env var, default 'low')
    """

    def __init__(
        self,
        name: str,
        prefix: str,
        api_key: str = None,
        model_name: str = None,
        response_model: Type[BaseModel] = None,
        prompt_template=None,
        memory_length: int = 100,
        env_params: dict = None,
        logger: logging.Logger = None,
        temperature: float = None,
        reasoning_effort: str = None,
    ):
        super().__init__(
            name=name,
            prefix=prefix,
            prompt_template=prompt_template,
            env_params=env_params,
            logger=logger,
        )

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model_name = model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
        self.temperature = (
            float(temperature)
            if temperature is not None
            else float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE))
        )
        self.reasoning_effort = reasoning_effort or os.getenv(
            "REASONING_EFFORT", DEFAULT_REASONING_EFFORT
        )
        # json_object response format is requested by default; automatically
        # disabled for the rest of the process if the model rejects it.
        self.use_response_format = (
            os.getenv("RESPONSE_FORMAT_JSON", "true").strip().lower() != "false"
        )
        self.response_model = response_model
        self.reponse_model_in_answer = self.__filter_in_answer_fields(response_model)
        self.prompt_template = prompt_template
        self.memory_length = memory_length
        self.logger = logger or logging.getLogger("experiment_logger")
        self.reponse_model_str = (
            "Respond only with a JSON object with this schema:\n{\n"
            + "\n".join(
                [
                    f'  "{name}": {field.annotation.__name__}'
                    for name, field in response_model.model_fields.items()
                    if field.json_schema_extra.get("in_answer", True)
                ]
            )
            + "\n}"
        )

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=OPENROUTER_BASE_URL,
        )

    async def act(self, prompt: str) -> Dict:
        """
        Execute pricing decision using the LLM API.

        Makes an API call to OpenRouter with retry logic and validates the
        response using the configured Pydantic model. Falls back to robust
        JSON extraction (markdown fences, first balanced JSON object) when
        the reply is not directly parseable.

        Args:
            prompt: Market context and decision prompt

        Returns:
            Dict containing agent name, validated response content, and
            per-call usage accounting (latency, tokens, cost) under "usage"

        Raises:
            ValidationError: If response validation fails after all retries
            Exception: If API call fails after all retries
        """
        for attempt in range(1, MAX_RETRIES + 1):
            self.logger.info(f"🔄 Attempt {attempt} for agent {self.name}")
            start_time = time.perf_counter()
            try:
                request_kwargs = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": self.reponse_model_str},
                        {"role": "system", "content": self.prefix},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self.temperature,
                    "extra_body": {
                        "reasoning": {"effort": self.reasoning_effort},
                        # Ask OpenRouter for usage accounting (token costs)
                        "usage": {"include": True},
                    },
                }
                if self.use_response_format:
                    request_kwargs["response_format"] = {"type": "json_object"}

                response = await self.client.chat.completions.create(**request_kwargs)
                latency = time.perf_counter() - start_time

                content = response.choices[0].message.content
                parse_failures = 0
                try:
                    parsed_json = json.loads(content)
                    if not isinstance(parsed_json, dict):
                        raise ValueError("Top-level JSON value is not an object")
                except (json.JSONDecodeError, ValueError) as e:
                    parse_failures = 1
                    self.logger.warning(
                        f"⚠️ Direct JSON parse failed for agent {self.name} "
                        f"(attempt {attempt}): {e}. Falling back to robust extraction."
                    )
                    try:
                        parsed_json = _extract_first_json_object(content)
                    except ValueError as extract_error:
                        self.logger.error(
                            f"❌ JSON extraction failed for agent {self.name} "
                            f"(attempt {attempt}): {extract_error}. "
                            f"Raw content (first 500 chars): {content[:500]!r}"
                        )
                        raise

                ## Validate using Pydantic
                validated = self.reponse_model_in_answer(**parsed_json)

                usage = self.__collect_usage(response, latency, attempt, parse_failures)
                self.logger.info(
                    f"✅ Agent {self.name}: latency {usage['latency_seconds']}s, "
                    f"prompt tokens {usage['prompt_tokens']}, "
                    f"completion tokens {usage['completion_tokens']}, "
                    f"reasoning tokens {usage['reasoning_tokens']}, "
                    f"cost {usage['cost_usd']} USD"
                )

                return {
                    "agent_name": self.name,
                    "content": validated.dict(),
                    "usage": usage,
                }

            except (ValidationError, ValueError) as e:
                self.logger.warning(
                    f"⚠️ Attempt {attempt} failed for agent {self.name} "
                    f"({type(e).__name__}): {str(e)[:300]}"
                )
                if attempt == MAX_RETRIES:
                    self.logger.error(
                        f"❌ Validation failed after {MAX_RETRIES} attempts for agent {self.name}: {str(e)}"
                    )
                    raise e
            except APIStatusError as e:
                if (
                    e.status_code == 400
                    and self.use_response_format
                    and "response_format" in str(e).lower()
                ):
                    self.use_response_format = False
                    self.logger.warning(
                        f"⚠️ Agent {self.name}: model rejected response_format "
                        f"({e}). Disabling it for the rest of this process and "
                        f"relying on robust JSON parsing."
                    )
                    continue
                self.logger.warning(
                    f"⚠️ Attempt {attempt} failed for agent {self.name} "
                    f"({type(e).__name__}): {str(e)[:300]}"
                )
                if attempt == MAX_RETRIES:
                    self.logger.error(
                        f"❌ Error after {MAX_RETRIES} attempts for agent {self.name}: {str(e)}"
                    )
                    raise e
            except Exception as e:
                self.logger.warning(
                    f"⚠️ Attempt {attempt} failed for agent {self.name} "
                    f"({type(e).__name__}): {str(e)[:300]}"
                )
                if attempt == MAX_RETRIES:
                    self.logger.error(
                        f"❌ Error after {MAX_RETRIES} attempts for agent {self.name}: {str(e)}"
                    )
                    raise e
            await asyncio.sleep(RETRY_DELAY_SECONDS * attempt)

    def __collect_usage(
        self, response, latency: float, attempt: int, parse_failures: int
    ) -> dict:
        """
        Extract per-call usage accounting from an OpenRouter response.

        Returns a dict with latency, prompt/completion/reasoning tokens,
        cost in USD (as reported by OpenRouter), the attempt number and the
        number of JSON parse failures for this call.
        """
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        reasoning_tokens = None
        completion_details = getattr(usage, "completion_tokens_details", None)
        if completion_details is not None:
            reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)
        # OpenRouter reports the cost inside usage when "usage": {"include": true}
        # is requested; fall back to a top-level attribute just in case.
        cost_usd = getattr(usage, "cost", None)
        if cost_usd is None:
            cost_usd = getattr(response, "cost", None)

        return {
            "latency_seconds": round(latency, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cost_usd": cost_usd,
            "attempt": attempt,
            "parse_failures": parse_failures,
        }

    def __filter_in_answer_fields(self, model: Type[BaseModel]) -> Type[BaseModel]:
        """
        Filter Pydantic model fields marked for inclusion in responses.

        Creates a new model containing only fields with 'in_answer=True'
        in their json_schema_extra metadata.

        Args:
            model: Source Pydantic model to filter

        Returns:
            New Pydantic model with only filtered fields
        """
        # Filter the fields that have 'in_answer=True'
        in_answer_fields = {
            name: (field.annotation, field.default)
            for name, field in model.model_fields.items()
            if field.json_schema_extra.get("in_answer", False)
        }
        # Create a new model pydantic with only the filtered fields
        return create_model(model.__name__ + "Filtered", **in_answer_fields)

    @property
    def type(self) -> str:
        """Return agent type identifier."""
        return "LLM_agent"
