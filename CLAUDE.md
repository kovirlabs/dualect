# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

dualect translates numeric code between Octave/MATLAB (`.m`) and Python/NumPy
(`.py`). It is a **tested knowledge base of translation rules** plus a small
deterministic engine, with an LLM handling what the engine can't yet, and every
translation verified by running both sides.

Read [docs/design.md](docs/design.md) before starting any work; it is the source of truth for
architecture, the rule format and the milestones.

## Current state

Design only. No package code exists yet. Next step is milestone 1 in
docs/design.md (skeleton, rule schema, verify harness, Octave in CI).

## Core ideas to preserve

- **One rule per construct or built-in**, in `rules/<category>/<rule-id>/`
  (`rule.yaml`, optional `translate.py`, `cases/*.m` + `*.py`). Keep rules small
  and independent so a language change touches one folder.
- **Status lifecycle**: `unsupported` → `guided` → `mechanical`. Only promote a
  rule to `mechanical` when its `translate.py` passes all its cases.
- **A behavior change in MATLAB/Octave is a new variant rule** with
  `since`/`until`, not an edit that breaks the old behavior.
- **Never translate silently wrong.** Unknown constructs raise a named
  "unsupported" error or go to the guided (LLM) path, and are reported.
- **Verify by execution**: outputs compared with `numpy.allclose`, Octave run via
  oct2py. A translation that wasn't verified must be labelled as such.
- **Output is idiomatic NumPy** a human would maintain, not a MATLAB emulation layer.
- `rule.yaml` `why`/`pitfalls` text is written for both a human learner and an
  LLM; keep it precise and short.

## Tooling (planned; match first-principles-code conventions)

Python 3.11+ managed with **uv**; ruff, mypy, pytest.

```bash
uv sync --all-groups      # install
uv run pytest             # tests (Octave-dependent tests skip if Octave is absent)
uv run ruff check . && uv run ruff format .
uv run mypy dualect
```

Octave is needed for execution tests (`brew install octave` on macOS).

## Conventions

- Keyword-only arguments for public functions; type-hint everything.
- Docstrings on every public class/function; comments explain *why*.
- Do not fork or copy code from SMOP/OMPC/LiberMate/mat2py; mining their test
  ideas is fine after checking the license.
- Never put "MATLAB" or "Octave" in package, module or command names (trademark /
  project-name confusion). Describing compatibility in prose is fine.
