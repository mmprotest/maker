# MAKER MDAP

This package implements the MAKER method (Maximal Agentic Decomposition with first-to-ahead-by-k voting and red flagging) along with a concrete Towers of Hanoi environment. It provides a CLI entry point `maker-hanoi` for solving or estimating performance on the puzzle using OpenAI-compatible language models.

## Installation

```bash
pip install -e .[test]
```

## Usage

Solve a Towers of Hanoi instance:

```bash
maker-hanoi solve --num-disks 4 --model gpt-4.1-mini
```

Estimate single-step success probability:

```bash
maker-hanoi estimate --num-disks 4 --num-samples 100 --model gpt-4.1-mini
```
