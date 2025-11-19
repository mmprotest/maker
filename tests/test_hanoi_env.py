import pytest

from maker_mdap.core import ParseError, ValidationError
from maker_mdap.hanoi import (
    TowersOfHanoiEnvironment,
    apply_move,
    is_goal_state,
    make_initial_state,
)


def test_initial_state_and_goal_detection():
    state = make_initial_state(3)
    assert state == [[3, 2, 1], [], []]
    assert not is_goal_state(state, 3)
    assert is_goal_state([[], [], [3, 2, 1]], 3)


def test_apply_move_enforces_rules():
    state = [[3, 2, 1], [], []]
    new_state = apply_move(state, [1, 0, 1])
    assert new_state == [[3, 2], [1], []]
    with pytest.raises(ValidationError):
        apply_move(state, [2, 2, 0])
    with pytest.raises(ValidationError):
        apply_move(new_state, [3, 1, 0])


def test_validate_move_accepts_optional_expected_move():
    env = TowersOfHanoiEnvironment(3)
    state = [[3, 2, 1], [], []]

    # Matching expected move: passes
    env._validate_move(state, [1, 0, 2], expected_move=[1, 0, 2])

    # Different but still legal move: accepted without raising
    env._validate_move(state, [1, 0, 1], expected_move=[1, 0, 2])

    # Illegal move still raises even if expected is provided
    with pytest.raises(ValidationError):
        env._validate_move(state, [3, 1, 0], expected_move=[1, 0, 2])


def test_parse_and_validate_response():
    env = TowersOfHanoiEnvironment(3)
    context_state = [[3, 2, 1], [], []]
    context = type("Ctx", (), {"step_index": 0, "state": context_state, "previous_action": None})
    raw = """
move = [1, 0, 2]
next_state = [[3, 2], [], [1]]
"""
    output = env.parse_and_validate_response(context, raw)
    assert output.action == [1, 0, 2]
    assert output.next_state == [[3, 2], [], [1]]

    fenced_raw = """
Here is the move:
```move = [1, 0, 2]```
And the state:
```next_state = [[3, 2], [], [1]]```
"""
    output = env.parse_and_validate_response(context, fenced_raw)
    assert output.action == [1, 0, 2]
    assert output.next_state == [[3, 2], [], [1]]

    inline_raw = """Narration before the move ```move = [1, 0, 2]``` and even more narration before declaring ```next_state = [[3, 2], [], [1]]```"""
    output = env.parse_and_validate_response(context, inline_raw)
    assert output.action == [1, 0, 2]
    assert output.next_state == [[3, 2], [], [1]]

    bad_raw = """
move = [2, 0, 1]
next_state = [[3, 1], [2], []]
"""
    with pytest.raises(ValidationError):
        env.parse_and_validate_response(context, illegal_move)

    malformed = "move [1,0,2]"
    output = env.parse_and_validate_response(context, malformed)
    assert output.action == [1, 0, 2]
    assert output.next_state == [[3, 2], [], [1]]

    alternate_valid_move = """
move = [1, 0, 1]
"""
    output = env.parse_and_validate_response(context, alternate_valid_move)
    assert output.action == [1, 0, 1]
    assert output.next_state == [[3, 2], [1], []]


def test_parse_accepts_json_and_colon_formats():
    env = TowersOfHanoiEnvironment(3)
    context_state = [[3, 2], [], [1]]
    context = type(
        "Ctx",
        (),
        {"step_index": 1, "state": context_state, "previous_action": [1, 0, 2]},
    )
    # Moving disk 1 again is legal but violates the enforced deterministic strategy
    raw = """
move = [1, 1, 2]
next_state = [[3, 2], [], [1]]
"""
    output = env.parse_and_validate_response(context, json_payload)
    assert output.action == [1, 0, 2]
    assert output.next_state == [[3, 2], [], [1]]

    colon_payload = """
Next action below:
move: [1, 0, 2]
next_state: [[3, 2], [], [1]]
"""
    output = env.parse_and_validate_response(context, colon_payload)
    assert output.action == [1, 0, 2]
    assert output.next_state == [[3, 2], [], [1]]

    move_only = """
move = [1, 0, 2]
"""
    output = env.parse_and_validate_response(context, move_only)
    assert output.action == [1, 0, 2]
    assert output.next_state == [[3, 2], [], [1]]


def test_deterministic_sequence_reaches_goal():
    for num_disks in (2, 3, 4):
        seq = compute_deterministic_sequence(num_disks)
        assert len(seq) == 2**num_disks - 1
        assert is_goal_state(seq[-1][2], num_disks)

