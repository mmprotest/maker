import itertools

from maker_mdap.config import MAKERConfig
from maker_mdap.core import (
    MAKERRunner,
    ParseError,
    SubtaskContext,
    SubtaskOutput,
    TaskEnvironment,
    ValidationError,
    run_voting_for_step,
)
from maker_mdap.llm_client import FakeLLMClient


class DummyEnv(TaskEnvironment):
    def __init__(self):
        self.state_value = 0

    def initial_state(self):
        return self.state_value

    def is_terminal(self, state, step_index):
        return step_index >= 1

    def make_prompts(self, context):
        return "sys", "user"

    def parse_and_validate_response(self, context, raw_response: str):
        if raw_response.startswith("BAD"):
            raise ParseError("bad")
        return SubtaskOutput(action=raw_response, next_state=context.state)

    def state_hash(self, state):
        return str(state)


class ProbabilisticResponder:
    def __init__(self, responses):
        self.responses = responses
        self.iterator = itertools.cycle(responses)

    def __call__(self, system_prompt, user_prompt):
        return next(self.iterator)


def test_first_to_ahead_by_k_voting_selects_majority():
    responses = ["A", "B", "A", "B", "A", "A"]
    env = DummyEnv()
    llm = FakeLLMClient(ProbabilisticResponder(responses))
    config = MAKERConfig(k=2, max_votes_per_step=10)
    context = SubtaskContext(step_index=0, state=0, previous_action=None)
    result_container = None
    output = run_voting_for_step(env, llm, config, context, result=result_container)
    assert output.action == "A"


def test_red_flags_retry_until_limit():
    # first two responses are too long, then a valid one
    long_text = "X" * 5000
    responses = [long_text, "BAD", "good"]
    env = DummyEnv()
    llm = FakeLLMClient(ProbabilisticResponder(responses))
    config = MAKERConfig(
        k=1,
        max_votes_per_step=5,
        max_red_flag_retries_per_step=3,
        max_response_characters_for_red_flag=100,
    )
    context = SubtaskContext(step_index=0, state=0, previous_action=None)
    output = run_voting_for_step(env, llm, config, context)
    assert output.action == "good"

