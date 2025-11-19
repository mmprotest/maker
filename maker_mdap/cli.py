"""Command line interface for MAKER Towers of Hanoi."""
from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Optional

import typer

from .config import LLMConfig, MAKERConfig
from .core import MAKERRunner, SubtaskContext, SubtaskOutput
from .hanoi import (
    TowersOfHanoiEnvironment,
    apply_move,
    compute_deterministic_sequence,
    is_goal_state,
    make_initial_state,
)
from .llm_client import LLMClient
from .logging_utils import configure_logging

app = typer.Typer(help="MAKER Towers of Hanoi CLI")


def _build_llm_config(model: str, api_key: Optional[str], base_url: Optional[str], temperature: float, max_output_tokens: int, request_timeout_seconds: int) -> LLMConfig:
    return LLMConfig(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        request_timeout_seconds=request_timeout_seconds,
    )


def _build_maker_config(k: int, max_response_chars: int, log_level: str) -> MAKERConfig:
    return MAKERConfig(
        k=k,
        max_response_characters_for_red_flag=max_response_chars,
        max_votes_per_step=50,
        max_red_flag_retries_per_step=100,
        log_level=log_level,
    )


@app.command()
def solve(
    num_disks: int = typer.Option(10, "--num-disks", "-d"),
    model: str = typer.Option("gpt-4.1-mini", "--model", "-m"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    k: int = typer.Option(3, "--k"),
    max_output_tokens: int = typer.Option(512, "--max-output-tokens"),
    max_response_chars: int = typer.Option(4000, "--max-response-chars"),
    log_level: str = typer.Option("INFO", "--log-level"),
    output_path: Optional[Path] = typer.Option(None, "--output-path"),
):
    """Solve a Towers of Hanoi instance using MAKER."""

    configure_logging(log_level)
    llm_config = _build_llm_config(model, api_key, base_url, temperature=0.1, max_output_tokens=max_output_tokens, request_timeout_seconds=60)
    maker_config = _build_maker_config(k, max_response_chars=max_response_chars, log_level=log_level)
    env = TowersOfHanoiEnvironment(num_disks=num_disks)
    llm = LLMClient(llm_config)
    runner = MAKERRunner(env=env, llm=llm, maker_config=maker_config)

    optimal_steps = 2 ** num_disks - 1
    typer.echo(f"Optimal number of steps: {optimal_steps}")
    outputs = runner.run_full_task()
    final_state = outputs[-1].next_state if outputs else env.initial_state()
    if not is_goal_state(final_state, num_disks):
        raise typer.Exit(code=1)

    typer.echo(f"Completed in {len(outputs)} steps")
    if runner.last_run_stats:
        votes = runner.last_run_stats.votes_per_step
        red_flags = runner.last_run_stats.red_flags_per_step
        if votes:
            typer.echo(
                f"Votes per step: min={min(votes)}, max={max(votes)}, mean={sum(votes)/len(votes):.2f}"
            )
        typer.echo(f"Total red flagged responses: {sum(red_flags)}")

    if output_path:
        actions = [output.action for output in outputs]
        output_path.write_text(json.dumps(actions))
        typer.echo(f"Saved moves to {output_path}")


@app.command()
def estimate(
    num_disks: int = typer.Option(10, "--num-disks", "-d"),
    model: str = typer.Option("gpt-4.1-mini", "--model", "-m"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    k: int = typer.Option(3, "--k"),
    max_output_tokens: int = typer.Option(512, "--max-output-tokens"),
    max_response_chars: int = typer.Option(4000, "--max-response-chars"),
    log_level: str = typer.Option("INFO", "--log-level"),
    num_samples: int = typer.Option(1000, "--num-samples"),
    use_red_flags: bool = typer.Option(True, "--use-red-flags/--no-red-flags"),
):
    """Estimate single-step accuracy without voting."""

    configure_logging(log_level)
    llm_config = _build_llm_config(model, api_key, base_url, temperature=0.1, max_output_tokens=max_output_tokens, request_timeout_seconds=60)
    maker_config = _build_maker_config(k, max_response_chars=max_response_chars, log_level=log_level)
    env = TowersOfHanoiEnvironment(num_disks=num_disks)
    llm = LLMClient(llm_config)

    sequence = compute_deterministic_sequence(num_disks)
    total_steps = len(sequence)
    sample_indices = random.sample(range(total_steps), k=min(num_samples, total_steps))
    successes = 0
    attempted = 0

    for idx in sample_indices:
        state, move, next_state = sequence[idx]
        prev_move = sequence[idx - 1][1] if idx > 0 else None
        context = SubtaskContext(step_index=idx, state=state, previous_action=prev_move)
        system_prompt, user_prompt = env.make_prompts(context)
        raw = llm.generate_step_response(system_prompt, user_prompt)
        try:
            parsed = env.parse_and_validate_response(context, raw)
        except Exception:
            if not use_red_flags:
                parsed = _forgiving_parse(raw)
                if parsed is None:
                    continue
                # validate against environment rules
                try:
                    env._validate_move(context.state, parsed.action)  # type: ignore[attr-defined]
                    env._validate_state(parsed.next_state)  # type: ignore[attr-defined]
                    expected = apply_move(context.state, parsed.action)
                    if expected != parsed.next_state:
                        continue
                except Exception:
                    continue
            else:
                continue
        attempted += 1
        if parsed.action == move and parsed.next_state == next_state:
            successes += 1

    p_hat = successes / attempted if attempted else 0.0
    typer.echo(f"Samples evaluated: {attempted}")
    typer.echo(f"Empirical single-step success rate p̂: {p_hat:.4f}")
    if attempted:
        recommended = "High reliability; k=3 likely sufficient." if p_hat > 0.995 else "Consider larger k or more voting."
        typer.echo(recommended)


def _forgiving_parse(raw_response: str) -> SubtaskOutput | None:
    """Attempt to parse loosely formatted responses."""

    move = None
    next_state = None
    for line in raw_response.splitlines():
        if "[" in line and "]" in line and "move" in line:
            try:
                move = ast.literal_eval(line[line.find("[") : line.rfind("]") + 1])
            except Exception:
                continue
        if "next_state" in line and "[[" in line:
            try:
                start = line.find("[[")
                end = line.rfind("]") + 1
                next_state = ast.literal_eval(line[start:end])
            except Exception:
                continue
    if move is not None and next_state is not None:
        return SubtaskOutput(action=move, next_state=next_state)
    return None


if __name__ == "__main__":
    app()

