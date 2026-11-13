# dualect

Translate numeric code between Octave/MATLAB-style `.m` files and Python/NumPy,
in both directions, with every translation **verified by running both sides**.

> **Status: early.** `dualect verify`, the rule tooling and the first 26 rules (Octave →
> Python, all `guided`, every case verified in both languages) work; scanning and
> translating are not built yet. See [docs/design.md](docs/design.md) section 11.

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
uv sync --all-groups                      # install (Python 3.11+, uv)
uv run dualect verify script.m script.py  # run both sides, compare every variable
uv run dualect rules list [--status guided]
uv run dualect rules new indexing/foo     # scaffold a rule folder
uv run pytest                             # validate rules, execute every case pair
```

`verify` needs GNU Octave (`brew install octave`, `apt install octave`).

Planned, not built yet:

```bash
dualect scan script.m                     # which rules does this file need?
dualect context script.m                  # LLM-ready context for those rules
dualect translate script.m -o script.py   # translate (scripted + LLM-guided)
```

## License

Apache-2.0. MATLAB is a registered trademark of The MathWorks, Inc. dualect is not
affiliated with MathWorks or the GNU Octave project.
