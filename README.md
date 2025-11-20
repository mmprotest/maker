# MAKER MDAP

MAKER MDAP implements the **Maximal Agentic Decomposition** method with first-to-ahead-by-`k` voting and red flagging. The package ships a Towers of Hanoi task environment and a `maker-hanoi` CLI for running the algorithm end-to-end against OpenAI-compatible language models.

## Project highlights

- **First-to-ahead-by-`k` voting** and red-flagging to reject unparseable or invalid LLM responses. The voting loop stops as soon as one candidate maintains a `k`-vote lead or the configured vote/red-flag limit is reached.
- **Towers of Hanoi environment** with prompt construction, strict move validation, and deterministic optimal sequences for evaluation runs.
- **Typed configuration models** for both the LLM client and MAKER runtime, with sensible defaults and environment variable overrides.
- **Typer-powered CLI** providing `solve` and `estimate` subcommands for running the full task or sampling single steps.

## Installation

Requirements: Python 3.11+. Create and activate a virtual environment before installing dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[test]
```

If you are using a non-default OpenAI-compatible endpoint, configure `OPENAI_API_KEY` and optionally `MAKER_BASE_URL` in your environment before running the CLI.

## Quickstart

1. Set your API key: `export OPENAI_API_KEY=sk-...` (and optionally `MAKER_BASE_URL=https://api.openai.com/v1`).
2. Install the package with the command above.
3. Run `maker-hanoi solve --num-disks 3 --model gpt-4.1-mini` to watch the full loop execute.
4. Inspect the generated move log to verify progress and voting behavior.

## Architecture overview

- **Core loop** (`maker_mdap.core`): orchestrates candidate generation, parsing, voting, and red-flag retries until a `k`-vote lead is achieved or limits are exceeded.
- **Task environments** (`maker_mdap.task`): define prompts, validation, and goal checking. The bundled Towers of Hanoi environment supplies deterministic optimal paths for evaluation.
- **Clients** (`maker_mdap.client`): wrap OpenAI-compatible chat endpoints with typed request/response handling, configurable temperatures, and token limits.
- **CLI** (`maker_mdap.cli`): Typer commands wire user input into the core loop and environment, handling configuration resolution from flags and environment variables.

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

### Configuration reference

The CLI builds configuration objects defined in `maker_mdap.config` and can also read settings from environment variables:

| Category | Env var | Default | Description |
| --- | --- | --- | --- |
| LLM | `MAKER_MODEL` | `gpt-4.1-mini` | Chat model name for candidate generation. |
|  | `OPENAI_API_KEY` | _required_ | Token used to authenticate requests. |
|  | `MAKER_BASE_URL` | OpenAI default | Override base URL for compatible providers. |
|  | `MAKER_TEMPERATURE` | `0.0` | Sampling temperature applied to generation requests. |
|  | `MAKER_MAX_OUTPUT_TOKENS` | `1024` | Token cap for each model completion. |
|  | `MAKER_REQUEST_TIMEOUT` | `30` | Timeout (seconds) applied to HTTP requests. |
| Runtime | `MAKER_K` | `3` | Vote lead required to accept a candidate. |
|  | `MAKER_MAX_VOTES_PER_STEP` | `10` | Upper bound on total votes per step before aborting. |
|  | `MAKER_MAX_RED_FLAG_RETRIES_PER_STEP` | `5` | Number of retries allowed after malformed responses. |
|  | `MAKER_MAX_RESPONSE_CHARS` | `4000` | Hard cutoff for parsing oversized model outputs. |
|  | `MAKER_LOG_LEVEL` | `INFO` | Logging verbosity for the CLI output. |

Configuration values are merged from defaults, environment variables, and CLI flags (highest precedence), making it easy to script sweeps of different models or `k` values.

### Sample outputs

- **Solve**: Prints each accepted move in `from->to` notation, followed by per-step vote counts and a final summary of total votes and red flags observed.
- **Estimate**: Reports the number of successful single-step predictions, the empirical accuracy, and a recommendation for a suitable `k` based on the observed distribution.

## Development

Run the test suite with:

```bash
pytest
```

Additional tips:

- Use `pytest -k hanoi` to focus on environment-specific behaviors.
- The CLI entry point is exposed as `maker-hanoi`; additional environments can implement the `TaskEnvironment` protocol in `maker_mdap.core` to plug into the existing voting loop.
- Type hints are enforced across the package; run `python -m maker_mdap.cli --help` to see the full command surface.
