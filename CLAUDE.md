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

Milestone 1 (skeleton) is done; next is milestone 2 in docs/design.md.

- `dualect/cli.py` — the `dualect` command. `verify` and `rules list|new` work;
  `scan`, `context`, `translate` exit 3 with "not implemented yet".
- `dualect/rulebook.py` loads rules; `dualect/validate.py` checks them against
  `dualect/schema/rule.schema.json` plus folder conventions; `dualect/scaffold.py`
  creates new rule folders (with `TODO` markers the tests reject).
- `dualect/runners.py` runs `.m` (oct2py, one reused session) and `.py` scripts;
  `dualect/compare.py` decides value equality; `dualect/verify.py` ties them
  together. Semantics are in docs/design.md section 12.1.
- `tests/` — `test_rules.py` validates every rule and executes every case pair
  (`@pytest.mark.octave`, skipped without Octave); `tests/data/beam.*` is the
  hand-written harness pair.
- CI: `.github/workflows/tests.yml` (Ubuntu apt Octave, `DUALECT_REQUIRE_OCTAVE=1`).

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

## Tooling (matches first-principles-code conventions)

Python 3.11+ managed with **uv**; ruff, mypy, pytest.

```bash
uv sync --all-groups      # install
uv run pytest             # tests (Octave-dependent tests skip if Octave is absent)
uv run pytest -m 'not octave'   # only the tests that need no Octave
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
