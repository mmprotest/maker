"""Towers of Hanoi environment and deterministic reference implementation."""
from __future__ import annotations

import ast
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


def _disk1_direction(num_disks: int) -> int:
    """Return step direction for disk 1 based on disk parity."""

    return -1 if num_disks % 2 == 1 else 1


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


def _disk1_clockwise_move(state: list[list[int]], num_disks: int) -> list[int]:
    for peg_idx, peg in enumerate(state):
        if peg and peg[-1] == 1:
            src = peg_idx
            break
    else:
        raise ValidationError("Disk 1 not found on any peg")
    target = (src + _disk1_direction(num_disks)) % 3
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
        return _disk1_clockwise_move(state, num_disks)
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
        disk1_rule = (
            "If the previous move did not move disk 1, move disk 1 counter-clockwise one peg (0 -> 2 -> 1 -> 0).\n"
            if self.num_disks % 2 == 1
            else "If the previous move did not move disk 1, move disk 1 clockwise one peg (0 -> 1 -> 2 -> 0).\n"
        )
        system_prompt = (
            "You are a precise agent solving Towers of Hanoi one move at a time.\n"
            "Pegs are 0-indexed (the leftmost peg is 0).\n"
            "Rules:\n"
            "- Only one disk can be moved at a time.\n"
            "- Only the top disk from any stack can be moved.\n"
            "- A larger disk may not be placed on top of a smaller disk.\n"
            "For all moves, follow the standard Tower of Hanoi procedure:\n"
            f"{disk1_rule}"
            "If the previous move did move disk 1, make the only legal move that does not involve moving disk 1.\n"
            "Use these clear steps to find the next move given the previous move and current state.\n"
            "Ensure your answer includes a single next move in this EXACT FORMAT:\n"
            "```move = [disk id, from peg, to peg]```\n"
            "Ensure your answer includes the next state resulting from applying the move to the current state in this EXACT FORMAT:\n"
            "```next_state = [[...], [...], [...]]```"
        )
        prev_move_str = (
            str(context.previous_action) if context.previous_action is not None else "<NONE>"
        )
        user_prompt = (
            "Previous move: {previous_move}\n"
            "Current State: {current_state}\n"
            "Based on the previous move and current state, find the single next move that follows the procedure and the resulting next state."
        ).format(previous_move=prev_move_str, current_state=context.state)
        return system_prompt, user_prompt

    def parse_and_validate_response(self, context: SubtaskContext, raw_response: str) -> SubtaskOutput:
        move_expr = self._extract_assignment_expr(raw_response, "move")
        state_expr = self._extract_assignment_expr(raw_response, "next_state")
        if move_expr is None or state_expr is None:
            raise ParseError("Missing move or next_state line")
        try:
            move = ast.literal_eval(move_expr)
            raw_state = ast.literal_eval(state_expr)
        except Exception as exc:  # noqa: BLE001
            raise ParseError(f"Failed to parse response: {exc}") from exc

        expected_move = _deterministic_next_move(
            context.state, context.previous_action, self.num_disks
        )
        self._validate_move(context.state, move, expected_move)
        expected_state = apply_move(context.state, move)
        next_state = self._normalize_state_representation(raw_state)
        self._validate_state(next_state)
        if expected_state != next_state:
            raise ValidationError("next_state does not match move applied to current state")

        return SubtaskOutput(action=move, next_state=next_state)

    def _normalize_state_representation(self, state: Any) -> list[list[int]]:
        if not isinstance(state, list) or len(state) != 3:
            raise ValidationError("State must be list of three lists")
        normalized: list[list[int]] = []
        for peg in state:
            normalized.append(self._normalize_peg(peg))
        return normalized

    def _normalize_peg(self, peg: Any) -> list[int]:
        if not isinstance(peg, list):
            raise ValidationError("Each peg must be a list")
        if any(not isinstance(disk, int) for disk in peg):
            raise ValidationError("All disks must be integers")
        if len(peg) <= 1:
            return peg.copy()
        if self._is_strictly_descending(peg):
            return peg.copy()
        if self._is_strictly_ascending(peg):
            return list(reversed(peg))
        raise ValidationError("Disks on peg must be in strictly descending order")

    @staticmethod
    def _is_strictly_descending(values: list[int]) -> bool:
        return all(values[i] > values[i + 1] for i in range(len(values) - 1))

    @staticmethod
    def _is_strictly_ascending(values: list[int]) -> bool:
        return all(values[i] < values[i + 1] for i in range(len(values) - 1))

    def _extract_assignment_expr(self, raw_response: str, key: str) -> str | None:
        prefix = f"{key} ="
        for line in raw_response.strip().splitlines():
            stripped = self._strip_code_fence(line.strip())
            if stripped.startswith(prefix):
                parts = stripped.split("=", 1)
                if len(parts) == 2:
                    expr = parts[1].strip()
                    if expr:
                        return expr
        return self._extract_assignment_expr_from_text(raw_response, prefix)

    @staticmethod
    def _strip_code_fence(line: str) -> str:
        if line.startswith("```") and line.endswith("```") and len(line) > 6:
            return line.strip("`").strip()
        return line

    def _extract_assignment_expr_from_text(self, raw: str, prefix: str) -> str | None:
        idx = raw.find(prefix)
        if idx == -1:
            return None
        idx += len(prefix)
        while idx < len(raw) and raw[idx] in " `:\t":
            idx += 1
        if idx >= len(raw):
            return None
        if raw[idx] in "[{":
            end = self._find_matching_bracket(raw, idx, raw[idx])
        else:
            end = idx
            while end < len(raw) and raw[end] not in "\r\n`":
                end += 1
        expr = raw[idx:end].strip()
        return expr or None

    @staticmethod
    def _find_matching_bracket(text: str, start: int, opening: str) -> int:
        closing = "]" if opening == "[" else "}"
        depth = 0
        for idx in range(start, len(text)):
            char = text[idx]
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return idx + 1
        return len(text)

    def _validate_move(
        self,
        state: list[list[int]],
        move: Any,
        expected_move: list[int] | None = None,
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
        if expected_move is not None and move != expected_move:
            raise ValidationError("Move does not follow deterministic strategy")

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

