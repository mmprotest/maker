"""Towers of Hanoi environment and deterministic reference implementation."""
from __future__ import annotations

import ast
import logging
import re
from typing import Any, Tuple

from .core import ParseError, SubtaskContext, SubtaskOutput, TaskEnvironment, ValidationError


def make_initial_state(num_disks: int) -> list[list[int]]:
    """Return the initial Towers of Hanoi state for ``num_disks`` disks."""

    if num_disks < 1:
        raise ValueError("num_disks must be positive")
    return [list(range(num_disks, 0, -1)), [], []]


def apply_move(state: list[list[int]], move: list[int]) -> list[list[int]]:
    """Apply a move to the given ``state`` and return the resulting state."""

    if not isinstance(move, list) or len(move) != 3 or not all(
        isinstance(x, int) for x in move
    ):
        raise ValidationError("Move must be a list of three integers")
    disk_id, source_peg, target_peg = move
    if source_peg == target_peg:
        raise ValidationError("Source and target pegs must differ")
    if source_peg not in (0, 1, 2) or target_peg not in (0, 1, 2):
        raise ValidationError("Peg indices must be 0,1,2")
    if not state[source_peg]:
        raise ValidationError("Source peg is empty")
    if state[source_peg][-1] != disk_id:
        raise ValidationError("Disk must be on top of source peg")
    if state[target_peg] and state[target_peg][-1] < disk_id:
        raise ValidationError("Cannot place larger disk on smaller disk")

    new_state = [peg.copy() for peg in state]
    new_state[source_peg].pop()
    new_state[target_peg].append(disk_id)
    return new_state


def is_goal_state(state: list[list[int]], num_disks: int) -> bool:
    """Check whether ``state`` is the goal configuration."""

    return state == [[], [], list(range(num_disks, 0, -1))]


def _clockwise_peg(peg: int) -> int:
    return (peg + 1) % 3


def _counterclockwise_peg(peg: int) -> int:
    return (peg + 2) % 3


def _legal_moves(state: list[list[int]]) -> list[Tuple[int, int, int]]:
    moves: list[Tuple[int, int, int]] = []
    for src in range(3):
        if not state[src]:
            continue
        disk = state[src][-1]
        for dst in range(3):
            if src == dst:
                continue
            if not state[dst] or state[dst][-1] > disk:
                moves.append((disk, src, dst))
    return moves


def _disk1_direction(num_disks: int) -> str:
    """Return the movement direction for disk 1 (clockwise/counterclockwise)."""

    return "clockwise" if num_disks % 2 == 0 else "counterclockwise"


def _disk1_directional_move(state: list[list[int]], num_disks: int) -> list[int]:
    for peg_idx, peg in enumerate(state):
        if peg and peg[-1] == 1:
            src = peg_idx
            break
    else:
        raise ValidationError("Disk 1 not found on any peg")
    direction = _disk1_direction(num_disks)
    target = _clockwise_peg(src) if direction == "clockwise" else _counterclockwise_peg(src)
    if state[target] and state[target][-1] < 1:
        raise ValidationError("Illegal placement for disk 1")
    return [1, src, target]


def _only_legal_move_excluding_disk1(state: list[list[int]]) -> list[int]:
    moves = [m for m in _legal_moves(state) if m[0] != 1]
    if len(moves) != 1:
        raise ValidationError("Expected exactly one legal move excluding disk 1")
    return list(moves[0])


def _deterministic_next_move(
    state: list[list[int]], previous_move: list[int] | None, num_disks: int
) -> list[int]:
    """Return the deterministic next move for the given ``state``.

    The strategy alternates moving disk 1 clockwise and making the only legal
    non-disk-1 move, which guarantees optimal progress toward the goal. This is
    pure state-based logic, so it can be used to both generate the reference
    solution and validate model outputs.
    """

    if previous_move is None or previous_move[0] != 1:
        return _disk1_directional_move(state, num_disks)
    return _only_legal_move_excluding_disk1(state)


def compute_deterministic_sequence(num_disks: int) -> list[Tuple[list[list[int]], list[int], list[list[int]]]]:
    """Compute the full optimal move sequence using the deterministic strategy."""

    state = make_initial_state(num_disks)
    previous_move: list[int] | None = None
    sequence: list[Tuple[list[list[int]], list[int], list[list[int]]]] = []
    total_steps = 2 ** num_disks - 1

    for _ in range(total_steps):
        move = _deterministic_next_move(state, previous_move, num_disks)
        next_state = apply_move(state, move)
        sequence.append((state, move, next_state))
        state = next_state
        previous_move = move

    return sequence


class TowersOfHanoiEnvironment(TaskEnvironment):
    """TaskEnvironment implementing the Towers of Hanoi for MAKER."""

    def __init__(self, num_disks: int):
        if num_disks < 1:
            raise ValueError("num_disks must be positive")
        self.num_disks = num_disks

    def initial_state(self) -> Any:
        return make_initial_state(self.num_disks)

    def is_terminal(self, state: Any, step_index: int) -> bool:
        return is_goal_state(state, self.num_disks)

    def state_hash(self, state: Any) -> str:
        return str(state)

    def make_prompts(self, context: SubtaskContext) -> Tuple[str, str]:
        system_prompt = (
            "You are a precise agent solving Towers of Hanoi one move at a time.\n"
            "Three pegs: 0, 1, 2. Disks: 1 is smallest, n is largest.\n"
            "Rules: move one disk at a time from the top of a peg to the top of another; never place a larger disk on a smaller one.\n"
            "Goal: move all disks from peg 0 to peg 2.\n"
            "Use the optimal iterative strategy: move disk 1 in the same direction every other turn (clockwise when the number of disks is even, counterclockwise when odd); between those moves, make the only legal move that does not use disk 1.\n"
            "Respond ONLY with a single JSON object, no prose or explanations, in the form:\n"
            '{"move": [disk_id, from_peg, to_peg], "next_state": [[...], [...], [...]]}\n'
            "The \"next_state\" field is optional; the system will compute it from your move."
        )
        prev_move_str = (
            str(context.previous_action) if context.previous_action is not None else "<NONE>"
        )
        user_prompt = (
            "Follow the rules above strictly.\n"
            f"Previous move: {prev_move_str}\n"
            f"Current state: {context.state}\n"
            "Output exactly:\n"
            "move = [disk_id, from_peg, to_peg]\n"
            "next_state = [[...], [...], [...]] (optional)"
        )
        return system_prompt, user_prompt

    def parse_and_validate_response(self, context: SubtaskContext, raw_response: str) -> SubtaskOutput:
        parsed = self._extract_move_and_state(raw_response)
        if parsed is None:
            raise ParseError("Missing move line")
        move, next_state = parsed

        expected_move = _deterministic_next_move(
            context.state, context.previous_action, self.num_disks
        )
        self._validate_move(context.state, move, expected_move)
        expected_state = apply_move(context.state, move)

        if next_state is not None:
            try:
                self._validate_state(next_state)
                if expected_state != next_state:
                    logger.debug(
                        "Model next_state mismatch; using computed state instead. expected=%s model=%s",
                        expected_state,
                        next_state,
                    )
            except ValidationError as exc:
                logger.debug("Ignoring invalid next_state from model: %s", exc)

        return SubtaskOutput(action=move, next_state=expected_state)

    def _extract_move_and_state(
        self, raw_response: str
    ) -> tuple[list[int], list[list[int]] | None] | None:
        """Parse model output tolerantly while enforcing structured content."""

        cleaned = raw_response.strip()
        cleaned = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", cleaned, flags=re.MULTILINE)

        # 1) Strict "move = ..." format
        move_line = None
        state_line = None
        for line in cleaned.splitlines():
            stripped = line.strip()
            if stripped.startswith("move ="):
                move_line = stripped.split("=", 1)[1].strip()
            if stripped.startswith("next_state ="):
                state_line = stripped.split("=", 1)[1].strip()
        if move_line:
            try:
                move = ast.literal_eval(move_line)
                next_state = ast.literal_eval(state_line) if state_line else None
                return move, next_state
            except Exception:  # noqa: BLE001
                pass

        # 2) JSON/dict style payload
        dict_like = None
        if cleaned.startswith("{") or cleaned.startswith("["):
            dict_like = cleaned
        else:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                dict_like = match.group(0)
        if dict_like:
            try:
                payload = ast.literal_eval(dict_like)
                if isinstance(payload, dict) and "move" in payload:
                    return payload["move"], payload.get("next_state")
            except Exception:  # noqa: BLE001
                pass

        # 3) Labeled lines with ":" delimiter
        move_match = re.search(r"move\s*:\s*(\[.*?\])", cleaned, re.DOTALL)
        state_match = re.search(r"next_state\s*:\s*(\[\s*\[.*?\]\s*,\s*\[.*?\]\s*,\s*\[.*?\]\s*\])", cleaned, re.DOTALL)
        if move_match:
            try:
                move = ast.literal_eval(move_match.group(1))
                next_state = ast.literal_eval(state_match.group(1)) if state_match else None
                return move, next_state
            except Exception:  # noqa: BLE001
                pass

        # 4) Fallback: first list of three ints + first triple-peg state
        move_match = re.search(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]", cleaned)
        state_match = re.search(
            r"\[\s*\[.*?\]\s*,\s*\[.*?\]\s*,\s*\[.*?\]\s*\]",
            cleaned,
            re.DOTALL,
        )
        if move_match:
            try:
                move = ast.literal_eval(move_match.group(0))
                next_state = (
                    ast.literal_eval(state_match.group(0)) if state_match else None
                )
                return move, next_state
            except Exception:  # noqa: BLE001
                pass

        return None

    def fallback_output(self, context: SubtaskContext) -> SubtaskOutput:
        """Return the deterministic next move when model output is unusable."""

        move = _deterministic_next_move(
            context.state, context.previous_action, self.num_disks
        )
        next_state = apply_move(context.state, move)
        return SubtaskOutput(action=move, next_state=next_state)

    def _validate_move(
        self, state: list[list[int]], move: Any, expected_move: list[int]
    ) -> None:
        if not isinstance(move, list) or len(move) != 3 or not all(
            isinstance(x, int) for x in move
        ):
            raise ValidationError("Move must be list of three integers")
        disk_id, source_peg, target_peg = move
        if disk_id < 1 or disk_id > self.num_disks:
            raise ValidationError("Invalid disk id")
        if source_peg not in (0, 1, 2) or target_peg not in (0, 1, 2):
            raise ValidationError("Peg indices must be 0,1,2")
        if source_peg == target_peg:
            raise ValidationError("Source and target cannot be the same")
        if not state[source_peg]:
            raise ValidationError("Source peg is empty")
        if state[source_peg][-1] != disk_id:
            raise ValidationError("Disk not on top of source peg")
        if move != expected_move:
            raise ValidationError("Move must follow deterministic optimal policy")

    def _validate_state(self, state: Any) -> None:
        if not isinstance(state, list) or len(state) != 3:
            raise ValidationError("State must be list of three lists")
        all_disks: list[int] = []
        for peg in state:
            if not isinstance(peg, list):
                raise ValidationError("Each peg must be a list")
            if any(not isinstance(d, int) for d in peg):
                raise ValidationError("All disks must be integers")
            if any(peg[i] <= peg[i + 1] for i in range(len(peg) - 1)):
                raise ValidationError("Disks on peg must be in strictly descending order")
            all_disks.extend(peg)
        expected = list(range(1, self.num_disks + 1))
        if sorted(all_disks) != expected:
            raise ValidationError("State must include each disk exactly once")

logger = logging.getLogger(__name__)

