from maker_mdap.config import MAKERConfig
from maker_mdap.core import MAKERRunner
from maker_mdap.hanoi import TowersOfHanoiEnvironment, compute_deterministic_sequence, is_goal_state
from maker_mdap.llm_client import FakeLLMClient


def test_end_to_end_runner_with_perfect_model():
    num_disks = 4
    env = TowersOfHanoiEnvironment(num_disks)
    sequence = compute_deterministic_sequence(num_disks)
    state_to_output = {str(state): (move, next_state) for state, move, next_state in sequence}

    def responder(system_prompt, user_prompt):
        for key, (move, next_state) in state_to_output.items():
            if key in user_prompt:
                return f"move = {move}\nnext_state = {next_state}"
        first_state, first_move, first_next = sequence[0]
        return f"move = {first_move}\nnext_state = {first_next}"

    llm = FakeLLMClient(responder)
    config = MAKERConfig(k=1, max_votes_per_step=5)
    runner = MAKERRunner(env, llm, config)
    outputs = runner.run_full_task()
    assert len(outputs) == 2 ** num_disks - 1
    final_state = outputs[-1].next_state
    assert is_goal_state(final_state, num_disks)

