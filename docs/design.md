# dualect — design

Status: 2026-11-12. This document is the plan; section 11 tracks what is built.

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

### 5.4 Translation (settled in milestone 4)

- **Representation.** Source scalars become Python numbers; a `1xN` row and an
  `Nx1` column both become 1-D arrays (the analysis remembers which one it was,
  so `c + r` gets `c[:, np.newaxis]` and `u' * v` becomes `np.outer(u, v)`);
  only a true matrix is 2-D; char arrays become `str`.
- **Per statement.** Every statement is planned from its construct tags: all
  of its rules `mechanical` -> translated by their `translate.py`; otherwise it
  becomes a *hole*, a `TODO(dualect)` comment carrying the source line and the
  reason, followed by `raise NotImplementedError`, so partial output can never
  run past an untranslated statement. A translator that meets a form it cannot
  translate correctly raises `Decline`, which also makes a hole (reported as
  `guided`). The report lists every statement with its rules, the status used
  and whether the file verified.
- **Translators** (`rules/<category>/<rule-id>/translate.py`, loaded by
  `dualect/translators.py`) define `translate(node, ctx)` and optionally
  `translate_store`, `translate_multi`, `position_outputs` and
  `translate_positions`. They build Python `ast` nodes through the `Context` of
  `dualect/emit.py`, which gives them facts (`ctx.info`), sub-expressions
  (`ctx.expr`), other mechanical rules' helpers (`ctx.module`) and the
  assumption channel (`ctx.assume`, emitted as `# dualect: assumed ...`).
- **Shared semantics in the engine**, not in any one rule: copy-on-write (an
  assignment `y = x`, a slice or a transpose gets `.copy()` when either side is
  later modified in place; a parameter the function modifies is copied on
  entry), and *position variables*: a variable that only ever holds positions
  (integer literals, `find`, the second output of `max`, a `for` range) and is
  only read inside subscripts is stored 0-based, so `i = 2; x(i + 1)` becomes
  `i = 1; x[i + 1]` and `for k = 1:n` over subscripts becomes `range(n)`. At
  script level such a variable gets a `# dualect: index` directive so `verify`
  compares it as 0-based.
- **Never silently wrong**, concretely: an element store into an array the
  translation holds as integers (`np.array([1, 2, 3])`) is declined unless the
  value is known to be an integer (NumPy would truncate); `max`/`min` of data
  not known to be NaN-free use `np.nanmax`/`np.fmax`; a `continue` inside
  `do ... until` is declined; indexing an array of unknown shape assumes a
  vector, and a product of unknown shapes assumes `@`, each with an `assumed`
  comment that verification then checks.
- **Hardening (settled after milestone 4 by adversarial probes,
  `tests/test_adversarial.py`).** A local function is analysed once for the
  join of its call sites, and so is a lambda: its body is annotated with the
  join of every call's argument facts, never the last call's. A row at one
  call and a column at another join to a vector of *unknown orientation*, and
  products, element-wise operations against a row/column, `size` and
  concatenation of such a vector are declined rather than assumed (each call
  needs a different `@`/broadcast). Logical operands are numbers in the source
  but NumPy keeps `bool + bool` as `bool`, so arithmetic on two logical
  operands converts the left one (`m.astype(int) + m`). NumPy integers wrap
  around where the source computes in doubles: `.^` on an integer array with
  an exponent above 2 or unknown is computed in floats (`x**40.0`, `2.0**x`),
  and `prod` of integers takes `dtype=float`. `mod`/`rem` by a known zero are
  declined (the source returns `x`), by an unknown divisor assumed nonzero.
  `for c = A` iterates `A.T.copy()` when the body writes into `c` (rows of
  `A.T` are views), and a loop over a range known to be empty is declined
  (the source leaves the variable as a `1x0` array). The normalizer turns
  `2.^x` into `2 .^x`: the grammar read it as the number `2.` and a matrix
  power. Horizontal concatenation of columns and matrices is
  `np.column_stack`.
- **Output.** ruff formats the file (its default style, so the result does not
  depend on the caller's configuration). The first line states the verdict:
  `verified against the .m source`, or `UNVERIFIED: <reason>` (holes, no
  Octave, `--no-verify`, or differing outputs). Exit codes: 0 verified,
  1 verification failed or `--strict` stopped (nothing written), 4 written but
  incomplete or unverified. Float literals keep their source spelling.
- **Layer 2 comparison** (section 7): translator output and the hand-written
  case are both printed back with `ast.unparse`, which removes formatting and
  comments; case comments are teaching notes, not code.

### 5.5 Python -> `.m` (settled in milestone 5)

- **One rule folder, two directions.** A rule covers Python constructs under
  `from_python: {status, match}` in its `rule.yaml`, next to the `.m` tags in
  `match`; its `translate.py` adds `lift(node, ctx)` (and `lift_store`,
  `lift_multi`, `lift_assign` where needed) beside `translate`. Status is
  tracked per direction: `status` is `.m` -> Python, `from_python.status` is
  Python -> `.m`, and a rule may be `mechanical` one way and `guided` the
  other. Python construct tags are `py:`-prefixed (`py:op:matmult`,
  `py:call:np.zeros`, `py:method:sum`, `py:subscript`, `py:for`, ...; the
  vocabulary is `dualect/pyconstructs.py`); a tag no rule claims is reported
  as unsupported by name (`py:call:np.fft.fft`, `py:stmt:try`,
  `py:import:scipy`).
- **Pipeline.** `dualect/pyinfer.py` infers facts over the Python `ast`
  (1-D vs 2-D, dtype, NaN-free, non-negative, may-be-a-view), because lifting
  depends on them. `dualect/lift.py` plans each statement exactly like the
  other direction (every construct must be `mechanical` from Python, else a
  hole: a `TODO(dualect)` comment and an `error(...)` call), and rules lift
  Python nodes into the shared IR, which already has `.m` semantics (1-based
  `Index`, inclusive `Range`, `mtimes`). `dualect/memit.py` only prints it:
  precedence, layout, comments, `1;` before functions, commas between matrix
  elements. `translate`, `scan` and `context` pick the direction from the
  suffix of the source file.
- **Representation.** A 1-D NumPy array is a `1xN` row on the `.m` side. So
  `A @ v` is `(A * v.').'`, `u @ v` is `u * v.'`, `np.outer(u, v)` is
  `u.' * v`, `A[:, j]` is `A(:, j+1).'`, `A.sum(axis=1)` is `sum(A, 2).'`,
  `for c in A.T` is `for c = A` with `c = c.'` first in the body.
- **Values stay Python's.** Indices are shifted where they are *used*
  (`x[i]` is `x(i + 1)`), so every variable holds the same value on both
  sides and verification needs no `index` directives; positions a function
  returns stay 0-based (`i = np.argmax(v)` is `[~, i] = max(v); i = i - 1`).
  The source's own `dualect:` directives are never copied; verifying a
  `.py` -> `.m` translation uses only the generated file's directives.
- **Never silently wrong, from Python.** Declined (a guided hole), because
  the `.m` side has no counterpart: writing through a NumPy alias or view
  (`y = x; y[0] = 1`); a `def` writing into an array argument without copying
  it first; a `def` reading a module-level variable; a lambda whose free
  variable changes after it is created (Python binds late, a handle captures);
  a variable that would hide a function the translation calls (`max = 3`);
  storing fractions into an integer array or numbers into a boolean array
  (NumPy truncates or converts); arithmetic on two boolean arrays (NumPy's
  `bool + bool` is OR); `max`/`argmax`/`np.maximum`/`np.clip` of data that
  may hold NaN (NumPy propagates it, the `.m` side skips it); `np.round` of
  fractions (halves to even); `np.linspace(a, b, 1)`; `and`/`or` whose value
  (an operand) is used outside a condition; `~` on integers (bitwise);
  2-D masks (row vs column order); a function that returns on some paths
  only. Assumed out loud (`% dualect: assumed ...`) and left to verification:
  a variable index or slice bound not known to be non-negative (it would count
  from the end in Python), `sqrt`/`log`/fractional powers of data not known to
  be non-negative (NaN vs complex), unknown divisors of `%`, `//`, `np.mod`.
- **Verification** runs the original `.py` and the generated `.m` and
  compares them as in section 12.1; the output's first line states the
  verdict (`% ... verified against the .py source` or `UNVERIFIED: ...`).

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
   matches (after formatting, comments ignored) the hand-written `.py`, and
   that code verifies by execution. A `guided` rule whose cases all pass this
   is flagged for promotion.
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

`scan`, `context` and `translate` take `--dialect`; `scan --json` gives
machine-readable output; `translate` also takes `--no-verify`.

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
4. **Deterministic translators** for the mechanical subset; combination corpus
   (done: section 5.4). 30 of 35 rules are `mechanical`: every case of each
   translates to the hand-written `.py` and verifies against Octave
   (`tests/test_translate.py`). Still `guided`: `functions/multiple-returns`
   (needs `nargout`-dependent bodies collapsed into one tuple return),
   `functions/nargin` (the `isempty` default pattern and `np.clip` need
   argument facts the analysis does not derive), `indexing/grow-on-assign`
   (the idiomatic list-append and `np.pad` rewrites span statements),
   `io/display` (printf formats) and `syntax/command-syntax` (its case needs
   `strsplit`). Three hand-written case files changed to the equally idiomatic
   form a statement-by-statement translator produces: `x = 1` (was `1.0`) in
   control-flow/while, `x**-1.0` on an integer array in operators/power, and
   line-by-line `z = x; z = z[~(z < 0)]` and `C = A.copy(); C[:] = ...` in
   indexing/logical and indexing/colon-all. `corpus/` holds three scripts
   (`tests/test_corpus.py`): `ode_rk4.m` translates completely and verifies;
   `beam_deflection.m` is 29/30 (its `fprintf` is guided) and
   `signal_filter.m` 29/33 (`filter` has no rule; two statements depend on it).
5. **Python → Octave** for the same subset via the shared IR (done: section
   5.5). Of the 35 rules, 29 are `mechanical` from Python (every `.py` case
   translates completely and verifies against its `.py` source; this includes
   `functions/multiple-returns`, still `guided` to Python), 3 are `guided`
   (`functions/nargin`, `indexing/grow-on-assign`, `io/display`) and 3 claim
   no Python construct (`builtins/isempty`, `syntax/line-continuation`,
   `syntax/command-syntax`). The corpus runs in
   reverse (`tests/test_reverse.py`): `ode_rk4` round-trips `.m` -> `.py` ->
   `.m` completely and verifies; the other two carry the holes of their
   forward translation. Before it, adversarial probes hardened the `.m` ->
   Python direction (section 5.4).
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
