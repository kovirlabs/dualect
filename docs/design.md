# dualect — design

Status: draft, 2026-11-12. This document is the plan; section 11 tracks what is built.

## 1. Problem

Engineers have a lot of numeric code written for Octave/MATLAB and want it in
Python (NumPy/SciPy), and sometimes need to go back. Existing tools fall short:

| Tool | Direction | Problem |
|---|---|---|
| SMOP, matlab2python, mat2py | `.m` → `.py` | Largely unmaintained; fragile hand-written parsers; output depends on a MATLAB-emulating runtime instead of idiomatic NumPy |
| OMPC, LiberMate | `.m` → `.py` | Abandoned |
| (none serious) | `.py` → `.m` | — |
| oct2py, Octave `pythonic` | runtime bridge | Calls across languages; does not translate code |
| General LLMs | both | Often right, sometimes subtly wrong, never verified, and they re-derive the same rules every time |

No tool translates arbitrary code in both directions, and none should claim to. The
semantics differ at the root:

- 1-based, end-inclusive ranges vs 0-based, end-exclusive slices
- `*` is matrix multiply in Octave; `@` in NumPy (`*` is element-wise)
- Octave arrays copy on write; NumPy arrays share views
- `end` inside an index, `nargin`/`nargout`, cell arrays, structs, command syntax
  (`hold on`), auto-growing arrays on out-of-range assignment

## 2. Approach in one paragraph

dualect is a **tested knowledge base of translation rules** with a growing
**deterministic core** and an **LLM for the rest**. Each language construct or
built-in function is one *rule*: a small, self-contained folder that documents
the Octave form, the Python form, why they differ, and paired examples that are
executed in both languages and compared. Rules that are dependable get a
deterministic translator; the rest are handed to an LLM as focused context. Every
translation, scripted or LLM-produced, is verified by running both sides.

## 3. Goals and non-goals

Goals
- Readable, idiomatic NumPy output a human would be happy to maintain.
- **Localized upkeep**: when a language feature changes, only the rule(s) for
  that feature change, and only their tests fail.
- **Verified output**: numeric equivalence checked by execution, not assumed.
- **LLM-ready knowledge**: an LLM given a file gets only the rules that file needs.
- Honest gaps: unsupported constructs raise a named error, never translate silently wrong.

Non-goals (for now)
- Full-language coverage, GUIs (GUIDE/App Designer), Simulink, toolboxes beyond
  a curated set of common functions, MEX, classdef OOP (later, maybe).
- Byte-for-byte round trips. A round trip should be *numerically* equivalent, not
  textually identical.

## 4. Rules

### 4.1 Layout

```
rules/
  <category>/<rule-id>/
    rule.yaml        # metadata + explanation (the knowledge)
    translate.py     # optional deterministic converter (the automation)
    cases/
      basic.m        # paired examples, executed and compared
      basic.py
      edge-empty.m
      edge-empty.py
```

Categories (initial): `syntax`, `indexing`, `operators`, `control-flow`,
`functions`, `builtins`, `types` (cells, structs, strings), `io`, `plotting`.

### 4.2 `rule.yaml`

```yaml
id: indexing/end-in-index
title: "`end` inside an index expression"
status: mechanical            # mechanical | guided | unsupported
dialects: [octave, matlab]
match: [index:end]            # construct tags this rule covers (section 5.3)
since: {octave: "3.0", matlab: "R2006a"}   # optional version range
until: null
octave: |
  last = x(end);
  tail = x(end-2:end);
python: |
  last = x[-1]
  tail = x[-3:]
why: >
  Octave's `end` resolves to the length of the indexed dimension and ranges are
  inclusive; Python uses negative indices and exclusive slice ends. `end-2:end`
  is three elements, so the Python start is -3, not -2.
pitfalls:
  - "`x(end+1) = v` grows the array in Octave; NumPy raises IndexError. See indexing/grow-on-assign."
  - "Inside nested indexing, `end` binds to the innermost array being indexed."
related: [indexing/one-based, indexing/ranges, indexing/grow-on-assign]
python_imports: []
```

The `why` and `pitfalls` fields are written for two readers at once: a human
learning the difference, and an LLM that will apply the rule.

### 4.3 Status lifecycle

```
unsupported ──► guided ──► mechanical
   (documented,   (LLM applies    (translate.py applies it;
    clear error)   rule + cases,   cases must pass in CI)
                   result verified)
```

- **mechanical**: `translate.py` exists and all cases pass. CI-enforced.
- **guided**: no converter (yet). The rule text and cases are given to an LLM;
  its output is verified by execution.
- **unsupported**: documented so the scanner can report it by name.

**Promotion loop.** When a guided translation passes verification, the
input/output pair can be saved as a new case (with human review). Once a rule has
enough passing cases to show a stable pattern, write `translate.py` and promote it
to `mechanical`. The deterministic share of the work grows with use.

### 4.4 Versions and dialects

MATLAB changes slowly and mostly additively (implicit expansion and `string` in
R2016b, `arguments` blocks in R2019b, `dictionary` in R2022b), and Octave differs
in places (`#` comments, `++`, `!=`, `endfunction`). A behavior change becomes a
**new variant rule** with its own `since`/`until`, not an edit to the old one. The
user picks a source dialect/version; the rule set is filtered accordingly.

## 5. Pipeline

```
 source file
     │
     ▼
 1. parse        .m  → tree-sitter-matlab   |  .py → Python stdlib `ast`
     ▼
 2. scan         walk the tree, tag each node with the rule id(s) it needs
     ▼
 3. plan         partition nodes: mechanical / guided / unsupported
     │            (unsupported → stop or report, per --strict)
     ▼
 4. translate    mechanical: run each rule's translate.py on its nodes
     │            guided:     build an LLM prompt from ONLY the needed rules
     ▼
 5. emit         assemble target source, add imports, format (ruff for Python)
     ▼
 6. verify       run source and target on the same inputs; compare outputs
                  (Octave via oct2py; numpy.allclose with per-rule tolerances)
     ▼
 report          per-node: rule id, status used, verified yes/no
```

### 5.1 Shared intermediate representation (IR)

Both parsers lower into one small IR, and both emitters read from it. The
language-specific rules (index base, operator meaning, copy semantics) live in the
lowering and lifting steps, so Python → Octave reuses the same rule knowledge
instead of being a second project.

The IR stays deliberately small: expressions, assignments, indexing (with explicit
base and inclusive/exclusive flags), calls, control flow, function definitions.
Anything the IR cannot express is, by definition, `unsupported` or `guided`.

### 5.2 Type and shape uncertainty

Some rules depend on information the source does not state (is `a * b` scalar ×
matrix or matrix × matrix?). Strategy, in order:
1. Infer from literals and known built-in return shapes.
2. Emit the safe form plus a `# dualect: assumed matrix multiply` comment.
3. Let verification catch the wrong guess.

Step 1 is `dualect/infer.py`: a flow-sensitive analysis over the IR that records,
for every expression, its shape class (scalar, row, column, matrix, empty, ...),
known dimensions, element type (real, complex, logical, char) and whether it is
known to be finite, non-negative or integer-valued. Branches are joined, loops
iterated to a fixed point, and each local function is analysed once for the
join of the argument facts at its call sites in the file.

### 5.3 Parsing, construct tags and scanning (settled in milestone 3)

- **Octave dialect.** `tree-sitter-matlab` knows only MATLAB syntax. Given
  Octave's extensions it reports errors (`#` comments, `#{ #}` blocks, `!=`,
  `!x`, `+=`) or, worse, misreads them silently (`do` and `until` parse as
  command syntax, `n++` as `n + (+ <missing>)`). `dualect/source.py` rewrites
  them, outside strings and comments, into forms the grammar reads (`#` -> `%`,
  `!=` -> `~=`, `endfunction` -> `end` + spaces, `do`/`until` -> marker
  `while`/assignment pairs, `x += e` and `x++` with the operator kept in a side
  table) and records each use, so `--dialect matlab` can reject them. No newline
  is added or removed and positions are mapped back to the original text.
- **Precedence.** The grammar nests some operator chains wrongly (`-2 ^ 2` as
  `(-2) ^ 2`, `a == b & c` as `a == (b & c)`, `a.^b'` as `a.^(b')`). The
  lowering flattens every operator chain and rebuilds it with the documented
  precedence (`2^-2^2` is `(2^-2)^2`, checked against Octave 11).
- **Names.** `f(x)` is an index when `f` is assigned anywhere in the enclosing
  function or script (or is a parameter/output), otherwise a call.
- **Construct tags.** The scanner tags each statement with what it uses:
  `op:mtimes`, `index:end`, `call:linspace`, `assign:grow`, ... (the vocabulary
  is `dualect/constructs.py`). Each `rule.yaml` lists the tags it covers in
  `match`, so a new built-in is a new folder with `match: [call:foo]` and no
  engine change. Two unversioned rules may not claim the same tag. Some tags
  need inferred facts: `index:logical` (a logical subscript) and `assign:grow`
  (a subscript of `end+k` or a constant beyond the known size, or a variable
  that is undefined or `[]`). A subscript that is merely *unknown* is not a
  grow: NumPy raises `IndexError` there, so a wrong guess is loud.
- **Gaps are named.** A tag no rule matches is reported as unsupported under
  that tag (`call:strsplit`, `expr:field`, `stmt:try`); text the parser could
  not read is `parse:error`; Octave-only syntax under `--dialect matlab` is
  `dialect:octave-only`.
- **Context.** `dualect context` prints a two-line header (the rules, and the
  constructs with no rule) followed by each needed rule's `rule.yaml` and up to
  two cases (`basic` first), as Markdown. Nothing else.

## 6. LLM integration

The repo is the LLM's memory: models do not retain anything between tasks, so the
knowledge must be cheap to fetch in exactly the right slice.

- `dualect scan file.m` lists the rule ids a file needs.
- `dualect context file.m` prints those rules (`rule.yaml` + one or two cases),
  nothing else; that is the prompt context.
- `llms.txt` (generated) indexes all rules with one line each.
- An MCP server (later) exposes `scan`, `get_rule`, `translate`, `verify` as tools,
  following the pattern used by first-principles-code's `fpc-mcp`.

LLM output is never trusted directly: it goes through step 6 (verify) like
everything else, and the report marks which nodes were LLM-translated.

## 7. Testing

Three layers, from cheap to realistic:

1. **Rule cases** (`rules/**/cases/`): each `.m`/`.py` pair runs in both
   languages and outputs are compared. A failure points at exactly one rule.
2. **Translator tests**: each `translate.py` on its own cases yields code that
   matches (after formatting) the hand-written `.py`.
3. **Combination corpus** (`corpus/`): realistic multi-construct snippets
   (a beam-deflection script, an ODE solve, a signal filter). Most real bugs are
   rule interactions (`end` + grow-on-assign + preallocation), and they show up
   here.

Octave is required for execution tests; CI installs it. Tests that need Octave are
marked so they skip cleanly on machines without it.

## 8. CLI

```bash
dualect scan script.m [--json]         # which rules does this file need?
dualect context script.m               # LLM-ready context for those rules
dualect translate script.m -o script.py [--strict] [--dialect matlab@R2023a]
dualect verify script.m script.py      # run both, compare
dualect rules list [--status guided]   # coverage at a glance
dualect rules new indexing/foo         # scaffold a rule folder
```

`translate` parses its arguments but exits with code 3 and a "not implemented
yet" message until milestone 4 lands. `scan` and `context` take `--dialect`;
`scan --json` gives machine-readable output.

## 9. Tech stack

- Python 3.11+ managed with uv; ruff, mypy, pytest (same conventions as
  first-principles-code).
- `tree-sitter` + `tree-sitter-matlab` (MIT, maintained) for parsing `.m`.
- stdlib `ast` for parsing Python.
- `oct2py` (MIT) to execute Octave from tests and `verify`.
- NumPy/SciPy as the translation target.
- PyYAML for rule files; a JSON Schema validates every `rule.yaml` in CI.

## 10. Prior art to mine (do not fork)

SMOP, matlab2python, mat2py, OMPC, LiberMate: useful for lists of supported
built-ins, test snippets and known edge cases. Check each license before copying
anything; prefer re-deriving cases.

NumPy's "NumPy for MATLAB users" page is the canonical human cheat-sheet and a
good seed for `builtins/*` rules (link, don't copy).

## 11. Milestones

1. **Skeleton** (done): package, CLI stub, rule schema + validator, rule scaffolder,
   Octave-in-CI. Verify harness runs one hand-written pair.
2. **First 20 rules** (done: the list below is 26 rules, all `guided`, 37 case
   pairs, each executed in both languages by `tests/test_rules.py`), Octave →
   Python, all with cases:
   - syntax: comments, line continuation, command vs function syntax
   - indexing: one-based, ranges, `end`, logical indexing, grow-on-assign, colon `(:)`
   - operators: `*` vs `.*`, `'` vs `.'`, `^` vs `.^`, `~=`/`!=`, `&&`/`||`
   - control-flow: `for` over matrix columns, `while`, `if/elseif`, `switch`
   - functions: definition, multiple returns, `nargin`
   - builtins: `zeros/ones/eye`, `linspace`, `size/numel/length`, `sum/mean/max`, `find`
3. **Scanner + context command** (done: section 5.3; 35 rules after adding
   `syntax/array-literals`, `types/char-strings`, `operators/arithmetic`,
   `operators/division`, `operators/augmented-assignment`,
   `builtins/elementwise-math`, `builtins/constants`, `builtins/isempty` and
   `io/display`, which the scanner showed every real file needs). The IR, the
   lowering and shape inference landed here because the scanner walks the IR.
4. **Deterministic translators** for the mechanical subset; combination corpus.
5. **Python → Octave** for the same subset via the shared IR.
6. **MCP server**, promotion-loop tooling, `llms.txt`.

## 12. Decisions

Settled by the owner on 2026-11-12 (these were the open questions of the first draft):

- **Target style**: plain NumPy only. No helper module, not even opt-in; a
  construct with no clean NumPy idiom is `guided` or `unsupported`, not emulated.
- **Octave in CI**: the Ubuntu LTS apt package (`apt-get install octave` on
  `ubuntu-24.04`), not a pinned container image. CI sets
  `DUALECT_REQUIRE_OCTAVE=1` so Octave tests fail instead of skipping there.
- **Verify tolerance**: a global default (`rtol=1e-9`, `atol=1e-12`, in
  `dualect/compare.py`) with a per-rule override in `rule.yaml`
  (`tolerance: {rtol: ..., atol: ...}`).
- **Where rules live**: in this repository only (a built wheel carries a copy so
  an installed `dualect` still finds them). No separate data package.
- **LLM calls**: dualect itself never calls an LLM API. The guided path only
  *produces context* (`dualect context`, later the MCP server); whichever LLM the
  user runs does the translating, and its output comes back through `verify`.

### 12.1 Verify semantics (milestone 1)

How `dualect verify a.m a.py` and the rule-case tests decide two scripts agree:

- A script's outputs are the variables it leaves in its workspace. The `.m`
  side runs via oct2py (`source`, in a cleared workspace); the `.py` side runs in
  a fresh namespace. `ans`, `_`-prefixed names, modules, classes and `def`
  functions are not outputs.
- Every `.m` output must exist in Python with an equal value. Python-only names
  are allowed (idiomatic code may need temporaries) and listed in the report. A
  pair that compares nothing fails.
- Values: `numpy.allclose` with NaN equal to NaN; singleton dimensions ignored
  (Octave's `1xN` equals a NumPy 1-D array) but other shapes must match; two
  empty arrays are equal; logical vs numeric is a mismatch; int vs float is not.
  Structs compare as dicts, cells as lists (column-major order). Function
  handles and lambdas only agree on *being* callable.
- Directives in comments of either file: `% dualect: index i j` (1-based in
  `.m`, 0-based in Python, compared as `m - 1`) and `% dualect: ignore k`
  (e.g. a loop variable whose final value legitimately differs). Unknown
  directives are errors, so a typo cannot silently weaken a check.
- `.m` case scripts that define functions start with `1;` and define them before
  use: Octave does not accept functions at the end of a script.
- A case file must not be named like a function it calls (`round.m`): the
  harness runs it from its own folder, where it would shadow the built-in.
