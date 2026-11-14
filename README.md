# dualect

Translate numeric code between Octave/MATLAB-style `.m` files and Python/NumPy,
in both directions, with every translation **verified by running both sides**.

> **Status: early.** Both directions work end to end: `.m` → Python with 30 of 35
> rules mechanical, Python → `.m` with 29 mechanical, every case verified by running
> both languages. See [docs/design.md](docs/design.md) section 11.

## Why

Existing translators are mostly abandoned, go one way only, and produce Python that
imitates MATLAB instead of idiomatic NumPy. LLMs can translate, but they re-derive
the same tricky rules every time and nothing checks the result.

dualect is built around **rules**: one small, tested folder per language construct
or built-in (indexing with `end`, `*` vs `.*`, `linspace`, ...). Dependable rules get
a scripted translator; the rest are handed to an LLM as focused context; every
result is executed and compared.

## Usage

```bash
uv sync --all-groups                             # install (Python 3.11+, uv)
uv run pytest                                    # rules, translators, corpus, both directions
uv run ruff check . && uv run ruff format --check .
uv run mypy dualect && uv run python tools/mypy_rules.py
```

### `.m` → Python

```bash
uv run dualect translate script.m -o script.py   # translate, then verify by running both
uv run dualect scan script.m                     # which rules does it need? what has none?
uv run dualect context script.m                  # those rules as LLM-ready prompt context
uv run dualect verify script.m script.py         # run both sides, compare every variable
```

`translate` handles each statement with the rules' deterministic translators. A
statement it cannot translate correctly is left as a `TODO(dualect)` hole that raises
`NotImplementedError`, and is reported; `dualect context` prints the rules an LLM
needs to fill it, and `--strict` refuses to write anything incomplete. The first line
of the output says whether it was verified against the `.m` source. Where a shape had
to be assumed the line says so (`# dualect: assumed ...`), and `# dualect: index k`
marks a variable stored 0-based because it only ever holds positions.

The `.m` side may use Octave syntax (`#` comments, `!=`, `++`, `+=`,
`do ... until`, `endfunction`); `--dialect matlab` reports it as Octave-only.

### Python → `.m`

```bash
uv run dualect translate model.py -o model.m     # the direction follows the suffix
uv run dualect scan model.py                     # rules needed, Python constructs with none
uv run dualect context model.py
```

The same rules translate back: a 1-D NumPy array becomes a `1xN` row, `A @ v`
becomes `(A * v.').'`, `x[i]` becomes `x(i + 1)` (variables keep their Python
values), and the output is checked by running the `.py` and the `.m`. Holes are
`% TODO(dualect)` comments followed by `error(...)`. Python behaviour with no `.m`
counterpart is declined, not imitated: writing through a NumPy view or alias, a
function writing into its array argument, a lambda that reads a variable changed
later, `max` of data that may hold NaN, and the like (design section 5.5).

### Rules

```bash
uv run dualect rules list [--status guided] [--direction from-python]
uv run dualect rules check [RULE_ID ...]         # does each rule earn its status, both ways?
uv run dualect rules new indexing/foo            # scaffold a rule folder
uv run dualect llms-txt -o llms.txt              # regenerate the one-line-per-rule index
```

`rules check` runs every rule's translator on its cases in each direction, compares
the result with the hand-written target and runs both sides; it fails when a rule
marked `mechanical` does not qualify, or a `guided` one does (promote it).
[`llms.txt`](llms.txt) lists every rule in one line for an LLM.

### MCP server

```bash
uv run dualect mcp                               # stdio; tools: scan, get_rule, translate, verify
```

A client configuration (Claude Code, Claude Desktop, ...) looks like:

```json
{"mcpServers": {"dualect": {"command": "uv",
  "args": ["run", "--directory", "/path/to/dualect", "dualect", "mcp"]}}}
```

The server gives the client's model the rules and the checks; dualect itself never
calls an LLM API, and anything the model writes can be checked with `verify`.

`translate`, `verify` and the execution tests need GNU Octave (`brew install octave`,
`apt install octave`).

## License

Apache-2.0. MATLAB is a registered trademark of The MathWorks, Inc. dualect is not
affiliated with MathWorks or the GNU Octave project.
