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

Milestones 1-4 are done; next is milestone 5 (Python -> Octave via the shared
IR) in docs/design.md section 11.

- `rules/` — 35 rules in 8 categories; 30 are `mechanical` (a `translate.py`
  whose output equals every case's hand-written `.py` and verifies against
  Octave), 5 `guided` (reasons in design.md section 11). Each `rule.yaml` lists
  the construct tags it covers in `match`. `.m` cases must run on Octave 8 (CI,
  apt) and 11 (local); `.py` cases are ruff-formatted idiomatic NumPy.
- `corpus/` — realistic multi-construct scripts run through the whole pipeline.
- `dualect/cli.py` — `scan`, `context`, `translate`, `verify`, `rules list|new`.
- Front end: `dualect/source.py` rewrites Octave-only syntax so the MATLAB-only
  tree-sitter grammar reads it; `dualect/parse.py` parses; `dualect/lower.py`
  lowers to the shared IR (`dualect/ir.py`), rebuilding operator precedence the
  grammar gets wrong; `dualect/infer.py` infers shapes and types;
  `dualect/constructs.py` tags constructs; `dualect/scan.py` maps tags to rules;
  `dualect/context.py` prints LLM context. Design notes: design.md 5.3.
- Back end: `dualect/emit.py` plans each statement (all rules mechanical, or a
  reported hole), dispatches to the rules' translators
  (`dualect/translators.py`) and owns copy semantics and 0-based position
  variables; `dualect/pipeline.py` formats (ruff), writes, verifies and
  reports; `dualect/pyformat.py` also does the layer-2 comparison. Design
  notes: design.md 5.4.
- `dualect/rulebook.py` loads rules; `dualect/validate.py` checks them against
  `dualect/schema/rule.schema.json` plus folder conventions; `dualect/scaffold.py`
  creates new rule folders (with `TODO` markers the tests reject).
- `dualect/runners.py` runs `.m` (oct2py, one reused session) and `.py` scripts;
  `dualect/compare.py` decides value equality; `dualect/verify.py` ties them
  together. Semantics are in docs/design.md section 12.1.
- `tests/` — `test_rules.py` (every case pair executed), `test_scan.py`
  (parsing, Octave dialect, precedence, scanning, context), `test_translate.py`
  (layer 2 and execution for every mechanical rule, pipeline behavior),
  `test_corpus.py` (layer 3). Octave tests are `@pytest.mark.octave`.
- CI: `.github/workflows/tests.yml` (Ubuntu apt Octave, `DUALECT_REQUIRE_OCTAVE=1`).

## Core ideas to preserve

- **One rule per construct or built-in**, in `rules/<category>/<rule-id>/`
  (`rule.yaml`, optional `translate.py`, `cases/*.m` + `*.py`). Keep rules small
  and independent so a language change touches one folder.
- **Status lifecycle**: `unsupported` → `guided` → `mechanical`. Only promote a
  rule to `mechanical` when its `translate.py` passes all its cases: output equal
  to the hand-written `.py` (comments ignored) *and* verified against Octave.
  A translator that cannot translate a form correctly raises `Decline` (the
  statement becomes a reported hole); a shape guess must go through
  `ctx.assume` so the output says so.
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
