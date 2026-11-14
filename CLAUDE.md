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

Milestones 1-6 of docs/design.md are done, plus a coverage pass (section 11,
item 7). Both directions work end to end:
`.m` -> Python (dualect/lower.py, infer.py, emit.py; 58 of 61 rules mechanical)
and Python -> `.m` (dualect/pyinfer.py, lift.py, memit.py; 54 mechanical), one
rule folder per construct serving both (`from_python:` in rule.yaml, `lift()` in
translate.py, status per direction; inference facts for a built-in come from
its rule's `infer_call`/`pyinfer_call`). Still guided: functions/nargout,
io/display, builtins/tic-toc. `dualect rules check` enforces that every
status is earned; `dualect mcp` serves scan/get_rule/translate/verify;
`llms.txt` is generated (`dualect llms-txt -o llms.txt`) and must stay current.
Verify compares printed output when a script carries `% dualect: stdout`
(translations that print add it) and random values by shape (`shape r`).
Adversarial probes live in tests/test_adversarial*.py and tests/test_reverse.py.
All six corpus scripts are tested; three translate completely both ways.
Next: cell arrays and structs, `cond`/`ind2sub`-style gaps the corpus shows.
The project site is `docs/index.html` (GitHub Pages from `/docs`, TagBrush.css);
keep its numbers in step with README.md and run TagCheck after editing it.

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
uv run python tools/mypy_rules.py   # mypy over rules/**/translate.py
uv run dualect rules check          # every rule earns its status, both directions
uv run dualect llms-txt -o llms.txt # after adding or changing a rule
```

Octave is needed for execution tests (`brew install octave` on macOS).

## Conventions

- Keyword-only arguments for public functions; type-hint everything.
- Docstrings on every public class/function; comments explain *why*.
- Do not fork or copy code from SMOP/OMPC/LiberMate/mat2py; mining their test
  ideas is fine after checking the license.
- Never put "MATLAB" or "Octave" in package, module or command names (trademark /
  project-name confusion). Describing compatibility in prose is fine.
