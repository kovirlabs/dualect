# dualect

Translate numeric code between Octave/MATLAB-style `.m` files and Python/NumPy,
in both directions, with every translation **verified by running both sides**.

> **Status: design phase.** Nothing is installable yet. See [docs/design.md](docs/design.md).

## Why

Existing translators are mostly abandoned, go one way only, and produce Python that
imitates MATLAB instead of idiomatic NumPy. LLMs can translate, but they re-derive
the same tricky rules every time and nothing checks the result.

dualect is built around **rules**: one small, tested folder per language construct
or built-in (indexing with `end`, `*` vs `.*`, `linspace`, ...). Dependable rules get
a scripted translator; the rest are handed to an LLM as focused context; every
result is executed and compared.

## Planned usage

```bash
dualect scan script.m                     # which rules does this file need?
dualect translate script.m -o script.py   # translate (scripted + LLM-guided)
dualect verify script.m script.py         # run both, compare outputs
```

## License

Apache-2.0. MATLAB is a registered trademark of The MathWorks, Inc. dualect is not
affiliated with MathWorks or the GNU Octave project.
