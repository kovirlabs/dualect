# dualect

Translate numeric code between Octave/MATLAB-style `.m` files and Python/NumPy,
in both directions, with every translation **verified by running both sides**.

> **Status: early.** Octave → Python works end to end: `scan`, `context`,
> `translate` and `verify`, with 35 rules, 30 of them translated mechanically and every
> case verified in both languages. Python → Octave is next. See
> [docs/design.md](docs/design.md) section 11.

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
uv run dualect translate script.m -o script.py   # translate, then verify by running both
uv run dualect scan script.m                     # which rules does it need? what has none?
uv run dualect context script.m                  # those rules as LLM-ready prompt context
uv run dualect verify script.m script.py         # run both sides, compare every variable
uv run dualect rules list [--status guided]
uv run dualect rules new indexing/foo            # scaffold a rule folder
uv run pytest                                    # rules, translators, corpus
```

`translate`, `verify` and the execution tests need GNU Octave (`brew install octave`,
`apt install octave`).

`translate` handles each statement with the rules' deterministic translators. A
statement it cannot translate correctly is left as a `TODO(dualect)` hole that raises
`NotImplementedError`, and is reported; `dualect context` prints the rules an LLM
needs to fill it, and `--strict` refuses to write anything incomplete. The first line
of the output says whether it was verified against the `.m` source. Where a shape had
to be assumed the line says so (`# dualect: assumed ...`), and `# dualect: index k`
marks a variable stored 0-based because it only ever holds positions.

The `.m` side may use Octave syntax (`#` comments, `!=`, `++`, `+=`,
`do ... until`, `endfunction`); `--dialect matlab` reports it as Octave-only.

## License

Apache-2.0. MATLAB is a registered trademark of The MathWorks, Inc. dualect is not
affiliated with MathWorks or the GNU Octave project.
