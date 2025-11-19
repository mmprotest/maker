# MAKER MDAP

MAKER MDAP implements the **Maximal Agentic Decomposition** method with first-to-ahead-by-`k` voting and red flagging. The package ships a Towers of Hanoi task environment and a `maker-hanoi` CLI for running the algorithm end-to-end against OpenAI-compatible language models.

## Project highlights

- **First-to-ahead-by-`k` voting** and red-flagging to reject unparseable or invalid LLM responses. The voting loop stops as soon as one candidate maintains a `k`-vote lead or the configured vote/red-flag limit is reached.
- **Towers of Hanoi environment** with prompt construction, strict move validation, and deterministic optimal sequences for evaluation runs.
- **Typed configuration models** for both the LLM client and MAKER runtime, with sensible defaults and environment variable overrides.
- **Typer-powered CLI** providing `solve` and `estimate` subcommands for running the full task or sampling single steps.

## Installation

Requirements: Python 3.11+

```bash
pip install -e .[test]
```

If you are using a non-default OpenAI-compatible endpoint, configure `OPENAI_API_KEY` and optionally `MAKER_BASE_URL` in your environment before running the CLI.

## Usage

### Solve a Towers of Hanoi instance

Runs the full MAKER loop until the puzzle reaches the goal state (or exits with a non-zero code if it cannot).

```bash
maker-hanoi solve \
  --num-disks 4 \
  --model gpt-4.1-mini \
  --k 3 \
  --max-output-tokens 512 \
  --max-response-chars 4000 \
  --log-level INFO
```

Useful flags:

- `--output-path /path/to/moves.json` saves the sequence of moves for replay or verification.
- `--base-url` and `--api-key` can override any environment defaults for the underlying LLM client.
- `--max-steps` (defaults to the optimal count) aborts early if the model fails to make progress.
- Use `--log-level INFO` or lower to stream each chosen move and resulting state as the solver runs.

On completion, the command prints vote and red-flag counts per step so you can gauge reliability.

### Estimate single-step accuracy

Draws random steps from the deterministic optimal sequence, queries the model once per step, and reports the empirical success rate along with a recommendation about whether `k=3` is sufficient.

```bash
maker-hanoi estimate --num-disks 4 --num-samples 100 --model gpt-4.1-mini
```

Set `--no-red-flags` to allow best-effort parsing of malformed responses (useful for debugging prompt quality).

## Configuration

The CLI builds configuration objects defined in `maker_mdap.config` and can also read settings from environment variables:

- LLM: `MAKER_MODEL`, `OPENAI_API_KEY`, `MAKER_BASE_URL`, `MAKER_TEMPERATURE`, `MAKER_MAX_OUTPUT_TOKENS`, `MAKER_REQUEST_TIMEOUT`
- MAKER runtime: `MAKER_K`, `MAKER_MAX_VOTES_PER_STEP`, `MAKER_MAX_RED_FLAG_RETRIES_PER_STEP`, `MAKER_MAX_RESPONSE_CHARS`, `MAKER_LOG_LEVEL`

## Development

Run the test suite with:

```bash
pytest
```

The CLI entry point is exposed as `maker-hanoi`; additional environments can implement the `TaskEnvironment` protocol in `maker_mdap.core` to plug into the existing voting loop.
