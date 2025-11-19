"""Core MDAP / MAKER algorithm components."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple

from .config import LLMConfig, MAKERConfig
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class MDAPError(Exception):
    """Base exception for MDAP errors."""


class ParseError(MDAPError):
    """Raised when a model response cannot be parsed."""


class ValidationError(MDAPError):
    """Raised when a parsed response fails environment validation."""


class RedFlagError(MDAPError):
    """Raised when a response is red flagged."""


class VoteLimitExceededError(MDAPError):
    """Raised when voting cannot produce a winner within configured limits."""


@dataclass
class SubtaskContext:
    """Context for a single subtask/step."""

    step_index: int
    state: Any
    previous_action: Any | None


@dataclass
class SubtaskOutput:
    """Output for a single subtask."""

    action: Any
    next_state: Any


@dataclass(frozen=True)
class VoteCandidateKey:
    """Key used for tallying votes for a candidate output."""

    key: str


def serialize_candidate(action: Any, next_state: Any) -> str:
    """Serialize a candidate to deterministic JSON string for vote counting."""

    return json.dumps({"action": action, "next_state": next_state}, sort_keys=True)


class TaskEnvironment(Protocol):
    """Abstract interface for MDAP environments."""

    def initial_state(self) -> Any: ...

    def is_terminal(self, state: Any, step_index: int) -> bool: ...

    def make_prompts(self, context: SubtaskContext) -> Tuple[str, str]: ...

    def parse_and_validate_response(self, context: SubtaskContext, raw_response: str) -> SubtaskOutput: ...

    def state_hash(self, state: Any) -> str: ...


class RedFlagger:
    """Red flagging logic to filter problematic responses."""

    def __init__(self, env: TaskEnvironment, maker_config: MAKERConfig):
        self.env = env
        self.maker_config = maker_config

    def get_valid_output_or_red_flag(
        self, context: SubtaskContext, raw_response: str
    ) -> tuple[SubtaskOutput | None, str | None]:
        if len(raw_response) > self.maker_config.max_response_characters_for_red_flag:
            reason = f"response too long ({len(raw_response)} chars)"
            logger.debug("Red flag: %s", reason)
            return None, reason
        try:
            return self.env.parse_and_validate_response(context, raw_response), None
        except (ParseError, ValidationError) as exc:
            reason = f"parse/validation error: {exc}"
            logger.debug("Red flag: %s", reason)
            return None, reason


class VotingResult:
    """Result of a voting round."""

    def __init__(self):
        self.vote_counts: Dict[str, int] = {}
        self.candidates: Dict[str, SubtaskOutput] = {}
        self.red_flags: int = 0
        self.total_votes: int = 0

    @property
    def total_attempts(self) -> int:
        """Total model calls made, including red-flagged responses."""

        return self.total_votes + self.red_flags

    def record_vote(self, key: str, output: SubtaskOutput) -> None:
        self.total_votes += 1
        self.vote_counts[key] = self.vote_counts.get(key, 0) + 1
        self.candidates.setdefault(key, output)

    def record_red_flag(self) -> None:
        self.red_flags += 1

    def leader(self) -> Tuple[str, int]:
        if not self.vote_counts:
            return "", 0
        leader_key = max(self.vote_counts.items(), key=lambda item: item[1])[0]
        return leader_key, self.vote_counts[leader_key]

    def next_highest(self, leader_key: str) -> int:
        filtered = [count for key, count in self.vote_counts.items() if key != leader_key]
        return max(filtered) if filtered else 0


@dataclass
class RunStats:
    """Aggregated statistics across a MAKER run."""

    votes_per_step: list[int]
    red_flags_per_step: list[int]
    attempts_per_step: list[int]


def run_voting_for_step(
    env: TaskEnvironment,
    llm: LLMClient,
    maker_config: MAKERConfig,
    context: SubtaskContext,
    k: int | None = None,
    temperature_first_vote: float = 0.0,
    temperature_subsequent_votes: float | None = None,
    result: VotingResult | None = None,
) -> SubtaskOutput:
    """Execute first-to-ahead-by-k voting for a single step."""

    k = maker_config.k if k is None else k
    red_flagger = RedFlagger(env, maker_config)
    voting = result if result is not None else VotingResult()

    while voting.total_attempts < maker_config.max_votes_per_step:
        system_prompt, user_prompt = env.make_prompts(context)
        temperature = (
            temperature_first_vote
            if voting.vote_counts == {}
            else (
                temperature_subsequent_votes
                if temperature_subsequent_votes is not None
                else None
            )
        )
        attempt_number = voting.total_attempts + 1
        raw_response = llm.generate_step_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=None,
        )
        logger.debug(
            "LLM call step=%d attempt=%d temp=%s response_raw=%s",
            context.step_index,
            attempt_number,
            temperature,
            raw_response,
        )
        output, red_flag_reason = red_flagger.get_valid_output_or_red_flag(
            context, raw_response
        )
        if output is None:
            voting.record_red_flag()
            logger.info(
                "Step %d attempt %d red-flagged: %s",
                context.step_index,
                voting.total_attempts,
                red_flag_reason,
            )
            if voting.red_flags > maker_config.max_red_flag_retries_per_step:
                fallback = getattr(env, "fallback_output", None)
                if callable(fallback):
                    logger.warning(
                        "Red-flag limit exceeded; using fallback output for step %d",
                        context.step_index,
                    )
                    return fallback(context)
                raise VoteLimitExceededError(
                    "Exceeded maximum red-flag retries for step"
                )
            continue
        candidate_key = serialize_candidate(output.action, output.next_state)
        voting.record_vote(candidate_key, output)
        leader_key, leader_votes = voting.leader()
        margin = leader_votes - voting.next_highest(leader_key)
        logger.info(
            "Step %d vote %d: leader margin %d",
            context.step_index,
            voting.total_votes,
            margin,
        )
        if margin >= k:
            logger.info(
                "Step %d winner after %d votes (red flags %d)",
                context.step_index,
                voting.total_votes,
                voting.red_flags,
            )
            return voting.candidates[leader_key]

    fallback = getattr(env, "fallback_output", None)
    if callable(fallback):
        logger.warning(
            "No winner found within vote limit; using fallback output for step %d",
            context.step_index,
        )
        return fallback(context)

    raise VoteLimitExceededError("No winner found within vote limit")


class MAKERRunner:
    """Run a full task using the MAKER algorithm."""

    def __init__(self, env: TaskEnvironment, llm: LLMClient, maker_config: MAKERConfig):
        self.env = env
        self.llm = llm
        self.maker_config = maker_config
        self.last_run_stats: RunStats | None = None

    def run_full_task(self, max_steps: int | None = None) -> list[SubtaskOutput]:
        state = self.env.initial_state()
        previous_action = None
        step_index = 0
        outputs: list[SubtaskOutput] = []
        stats = RunStats(votes_per_step=[], red_flags_per_step=[], attempts_per_step=[])

        while not self.env.is_terminal(state, step_index):
            if max_steps is not None and step_index >= max_steps:
                raise MDAPError("Reached maximum allowed steps without termination")
            context = SubtaskContext(
                step_index=step_index,
                state=state,
                previous_action=previous_action,
            )
            voting_result = VotingResult()
            output = run_voting_for_step(
                env=self.env,
                llm=self.llm,
                maker_config=self.maker_config,
                context=context,
                result=voting_result,
            )
            outputs.append(output)
            stats.votes_per_step.append(voting_result.total_votes)
            stats.red_flags_per_step.append(voting_result.red_flags)
            stats.attempts_per_step.append(voting_result.total_attempts)
            logger.info(
                "Step %d: move %s -> state %s",
                step_index + 1,
                output.action,
                output.next_state,
            )
            state = output.next_state
            previous_action = output.action
            step_index += 1

        self.last_run_stats = stats
        return outputs

    @staticmethod
    def outputs_to_actions(outputs: list[SubtaskOutput]) -> list[Any]:
        return [o.action for o in outputs]

