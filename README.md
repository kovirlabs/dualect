# dualect

dualect translates numeric code between `.m` files (Octave/MATLAB style) and
Python/NumPy, in both directions, and checks every translation by running both
sides and comparing the results. It is a tested knowledge base of translation
*rules*, one small folder per language construct or built-in function, plus a
deterministic engine that applies the reliable ones. Statements it cannot
translate with confidence are left as marked holes and reported, never guessed;
an LLM can fill them using the same rules as context, and its output is checked
the same way.

Site: https://kovirlabs.github.io/dualect/ · Design: [docs/design.md](docs/design.md)

> **Status: early (0.1.0).** 61 rules. `.m` → Python: 58 mechanical, 3 guided.
> Python → `.m`: 54 mechanical, 1 guided. Details under [Status and limits](#status-and-limits).

## Example

[`corpus/ode_rk4.m`](corpus/ode_rk4.m) integrates a damped oscillator with
Runge-Kutta. An excerpt of the input:

```matlab
t = 0:h:t_end;
n = numel(t);
y = zeros(2, n);
y(:, 1) = [1; 0];   % x(0) = 1, x'(0) = 0

for k = 1:n-1
  k1 = f(t(k), y(:, k));
  ...
err = max(abs(y(1, :) - x_exact));
```

and the corresponding part of what `dualect translate corpus/ode_rk4.m -o ode_rk4.py`
writes:

```python
# Translated from ode_rk4.m by dualect; verified against the .m source.
import numpy as np

...
t = np.linspace(0, t_end, 301)
n = t.size
y = np.zeros((2, n))
y[:, 0] = np.array([1, 0])  # x(0) = 1, x'(0) = 0

# dualect: index k
for k in range(n - 1):
    k1 = f(t[k], y[:, k])
    ...
err = np.nanmax(np.abs(y[0, :] - x_exact))
```

The first line is the verdict: both files were run and all 18 variables matched.
`max` became `np.nanmax` because the `.m` `max` skips NaN; `# dualect: index k`
tells the checker that `k` holds a position and is 0-based on the Python side.

## Install

dualect needs Python 3.11+, [uv](https://docs.astral.sh/uv/), and GNU Octave for
anything that runs `.m` code (`translate` with verification, `verify`, and the
execution tests). It is not on PyPI yet.

```bash
brew install octave        # or: sudo apt install octave
git clone https://github.com/kovirlabs/dualect.git && cd dualect
uv sync --all-groups
uv run dualect --version
```

## Quick start

The direction follows the input file's suffix.

**`.m` → Python**

```bash
uv run dualect scan corpus/ode_rk4.m                     # rules it needs; constructs with none
uv run dualect translate corpus/ode_rk4.m -o ode_rk4.py  # translate, then run both and compare
```

**Python → `.m`.** Given `trapezoid.py`:

```python
import numpy as np

x = np.linspace(0, np.pi, 101)
y = np.sin(x)
area = 0.0
for k in range(x.size - 1):
    area = area + (x[k + 1] - x[k]) * (y[k] + y[k + 1]) / 2
err = abs(area - 2)
```

`uv run dualect translate trapezoid.py -o trapezoid.m` writes:

```matlab
% Translated from trapezoid.py by dualect; verified against the .py source.
x = linspace(0, pi, 101);
y = sin(x);
area = 0.0;
for k = 0:numel(x) - 2
  area = area + (x(k + 2) - x(k + 1)) * (y(k + 1) + y(k + 2)) / 2;
end
err = abs(area - 2);
```

Variables keep their Python values (`k` still counts from 0), so indices add 1.
Python behaviour with no `.m` counterpart is declined rather than imitated:
writing through a NumPy view, a function that modifies its array argument, a
lambda that reads a variable changed later, `max`/`argmax` over data that may
hold NaN, and similar ([design section 5.5](docs/design.md)).

The `.m` side may use Octave syntax (`#` comments, `!=`, `++`, `+=`,
`do ... until`, `endfunction`); `--dialect matlab` reports it as Octave-only.

## Reading the report

`translate` prints one line per statement: line and column, kind, the status
used, whether it was verified, and the rules it used. Under a statement it could
not translate, it prints why, for example `needs call:cond (no rule)` or
`translator declined: max of a value whose shape is unknown or along a singleton dimension`. Then come a
summary (`30 statements: 29 mechanical, 1 guided, 0 unsupported`), the
per-variable comparison when verification ran, and a `verified:` line.

An untranslated statement becomes a hole in the output: in Python, a
`# TODO(dualect)` comment quoting the source line followed by
`raise NotImplementedError(...)`; in `.m`, a `% TODO(dualect)` comment followed
by `error(...)`. The output's first line then says `UNVERIFIED` and why.
`# dualect: assumed ...` marks an assumption the engine could not prove (for
example a real `sqrt` argument). A translation that prints carries
`# dualect: stdout`, so verification also compares the printed text exactly;
`# dualect: shape r` compares random values by shape only.

| Exit code | `translate` | `verify` |
|---|---|---|
| 0 | complete and verified | every variable matches |
| 1 | verification found a difference, or `--strict` refused to write | a variable differs or is missing |
| 4 | written but unverified: holes remain, or `--no-verify` was given | — |

`--strict` writes nothing if any statement would be left as a hole.

## Commands

```bash
uv run dualect translate FILE -o OUT [--strict] [--no-verify] [--dialect octave|matlab@R2023a]
uv run dualect scan FILE [--json]       # rules a file needs, and constructs with none
uv run dualect context FILE             # those rules and their cases, as LLM prompt context
uv run dualect verify FILE.m FILE.py    # run both, compare every variable
                                        #   [--rtol] [--atol] [--index NAME] [--ignore NAME]
uv run dualect rules list [--status guided] [--direction from-python]
uv run dualect rules check [RULE_ID ...] # does each rule earn its status, both ways?
uv run dualect rules new CATEGORY/NAME  # scaffold a rule folder
uv run dualect llms-txt -o llms.txt     # regenerate the one-line-per-rule index
uv run dualect mcp                      # MCP server on stdio
```

## Rules

Each rule is a folder `rules/<category>/<rule-id>/`:

- `rule.yaml`: the construct in each language, `why` they differ and the
  `pitfalls`, written for a person learning the difference and for an LLM
  applying it; the construct tags it `match`es; its status per direction.
- `translate.py` (optional): the deterministic translator, for one or both directions.
- `cases/*.m` + `*.py`: paired examples, run in both languages and compared.

A rule's status, per direction, is `unsupported` (documented so it can be
reported by name), `guided` (an LLM applies it; the result is verified) or
`mechanical` (its `translate.py` translates every case, and each verifies).
`rules check` fails when a `mechanical` rule no longer qualifies, or when a
`guided` one qualifies and should be promoted. A behaviour that changed between
language versions gets a new rule with `since`/`until`, not an edit to the old one.

To add a rule:

1. `uv run dualect rules new indexing/my-rule`, then fill in `rule.yaml` and
   `cases/basic.m` + `cases/basic.py`, removing every `TODO`.
2. `uv run pytest tests/test_rules.py` runs the cases in both languages.
3. Optionally write `translate.py` (see existing rules such as
   [`rules/builtins/linspace`](rules/builtins/linspace)) and set the status to
   `mechanical` once `uv run dualect rules check indexing/my-rule` passes.
4. `uv run dualect llms-txt -o llms.txt` (a test fails when it is stale).

The rule format and the pipeline are specified in [docs/design.md](docs/design.md)
sections 4 and 5.

## Using it with an LLM

dualect never calls an LLM API itself; it supplies the knowledge and the check.
`dualect context FILE` prints only the rules that file needs, with verified case
pairs; the LLM fills the holes; `dualect verify` checks the result.
[`llms.txt`](llms.txt) lists every rule in one line.

`dualect mcp` serves the tools `scan`, `get_rule`, `translate` and `verify` over
the Model Context Protocol (stdio). A client configuration (Claude Code, Claude
Desktop, ...) looks like:

```json
{"mcpServers": {"dualect": {"command": "uv",
  "args": ["run", "--directory", "/path/to/dualect", "dualect", "mcp"]}}}
```

## Status and limits

From `uv run dualect rules list`: 61 rules with 88 case pairs.

- `.m` → Python: 58 mechanical; guided: `functions/nargout` (a body that
  depends on how many outputs were requested), `io/display` (a missing `;`:
  the display format depends on the value) and `builtins/tic-toc` (timing).
- Python → `.m`: 54 mechanical; guided: `builtins/tic-toc`; 6 rules cover
  `.m`-only constructs.

Built-ins with rules include printing (`printf`, `fprintf`, `sprintf`, `disp`,
`num2str`), `filter` (as `scipy.signal.lfilter`), reductions and statistics
(`sum mean max min prod cumsum cumprod diff any all std var median`), ordering
(`sort` with its index output, `unique`, `flip`), shape (`reshape repmat cat
horzcat vertcat meshgrid`), linear algebra (`norm dot cross kron inv det trace
rank`, `\` for square and tall systems), `interp1`, `polyfit`, `polyval`,
`trapz`, `rand`/`randn` (compared by shape only: no two runs share values) and
the element-wise math functions; `llms.txt` lists them all.
[NumPy for MATLAB users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)
served as the checklist. Forms a rule cannot translate exactly are declined
by name (for example `%d` of a fraction, `unique` with several outputs,
`interp1` with `'spline'`).

The [`corpus/`](corpus) scripts:

| Script | `.m` → Python | Verified | Back to `.m` |
|---|---|---|---|
| `ode_rk4.m` | 23/23 | yes | 21/21, verified |
| `beam_deflection.m` | 30/30 | yes | 27/27, verified |
| `signal_filter.m` | 33/33 | yes | 29/29, verified |
| `least_squares_fit.m` | 26/27 (`cond` has no rule) | no | holes carried over |
| `data_statistics.m` | 27/28 (`round` of an expression is declined) | no | holes carried over |
| `grid_heat.m` | 20/26 (`trapz` along a dimension, `ind2sub`, shapes lost after a slice) | no | holes carried over |

Not supported yet: cell arrays, structs, plotting, file I/O and built-ins
beyond the rules listed in `llms.txt`. Out of scope for now:
`classdef`, GUIs, Simulink, MEX and toolboxes. Verification shows the two
programs agree when run as written, not on every possible input.

## Development

```bash
uv sync --all-groups
uv run pytest                            # Octave tests skip if Octave is absent
uv run ruff check . && uv run ruff format --check .
uv run mypy dualect && uv run python tools/mypy_rules.py
```

CI installs Octave and sets `DUALECT_REQUIRE_OCTAVE=1`, so Octave tests fail
there instead of skipping. The project site is the single page
[`docs/index.html`](docs/index.html), served by GitHub Pages from `/docs`.

## License

Apache-2.0. MATLAB is a registered trademark of The MathWorks, Inc. dualect is not
affiliated with MathWorks or the GNU Octave project.
