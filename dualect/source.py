# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Rewriting Octave-only syntax so the tree-sitter grammar can read it.

The ``tree-sitter-matlab`` grammar knows only the MATLAB dialect. Fed Octave's
extensions it either reports an error (``#`` comments, ``!=``) or, worse,
*silently misreads* them (``do`` and ``until`` parse as command syntax, ``n++``
as ``n + (+ <missing>)``). Rather than fork the grammar, :func:`normalize`
rewrites those spellings into equivalents the grammar understands, outside
strings and comments, and records what it changed:

* ``#`` comments and ``#{ ... #}`` blocks become ``%`` (same length);
* ``!=`` becomes ``~=`` and ``!x`` becomes ``~x`` (same length);
* ``endfunction``, ``endif``, ``endfor``, ``endwhile``, ``endswitch``, ... become
  ``end`` padded with spaces (same length);
* ``x += e`` (and ``-=``, ``*=``, ``/=``, ``^=``) blank the operator character and
  record it in :attr:`NormalizedSource.aug_ops`;
* ``x++`` / ``x--`` / ``++x`` / ``--x`` statements become ``x+0`` / ``0+x`` (a
  blanked operator would read as command syntax) and are recorded in
  :attr:`NormalizedSource.increments`;
* ``do ... until cond`` becomes ``while __dualect_do__ ... __dualect_until__ =
  cond; end``, which :mod:`dualect.lower` turns back into a ``DoUntil`` node.

No newline is ever added or removed, so line numbers survive unchanged; the few
rewrites that change a line's length are mapped back by
:meth:`NormalizedSource.position`. Each rewrite is also recorded as an
:class:`OctaveFeature`, so a run with ``--dialect matlab`` can reject code that
is not valid MATLAB.
"""

from dataclasses import dataclass, field

DO_MARKER = "__dualect_do__"
UNTIL_MARKER = "__dualect_until__"

#: Octave's specific block terminators; all mean ``end``.
_END_KEYWORDS = frozenset(
    {
        "endfunction", "endif", "endfor", "endwhile", "endswitch", "end_try_catch",
        "end_unwind_protect", "endparfor",
    }
)  # fmt: skip

_AUG_OPS = {"+": "plus", "-": "minus", "*": "mtimes", "/": "mrdivide", "^": "mpower"}
_IDENT_START = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_")
_IDENT_CHARS = _IDENT_START | frozenset("0123456789")
#: A quote right after one of these characters is a transpose, not a string.
_TRANSPOSE_AFTER = _IDENT_CHARS | frozenset(")]}'.")
_OPEN = frozenset("([{")
_CLOSE = frozenset(")]}")


@dataclass(frozen=True)
class OctaveFeature:
    """One use of syntax that only Octave accepts.

    :param name: what it is, e.g. ``"# comment"`` or ``"!="``.
    :param line: 1-based line in the original source.
    :param col: 1-based column in the original source.
    """

    name: str
    line: int
    col: int


@dataclass
class NormalizedSource:
    """Source text after :func:`normalize`, with the way back to the original.

    :param original: the text as the user wrote it.
    :param text: the rewritten text handed to the parser.
    :param features: Octave-only syntax that was found.
    :param aug_ops: normalized offset of an ``=`` that was ``op=`` -> operator name.
    :param increments: normalized offset of the identifier of an ``x++``-style
        statement -> ``"plus"`` or ``"minus"``.
    """

    original: str
    text: str
    _to_original: list[int] = field(repr=False)
    features: list[OctaveFeature] = field(default_factory=list)
    aug_ops: dict[int, str] = field(default_factory=dict)
    increments: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Index line starts of the original text for :meth:`position`."""
        self._line_starts = [0]
        for index, char in enumerate(self.original):
            if char == "\n":
                self._line_starts.append(index + 1)

    def original_offset(self, offset: int) -> int:
        """Map a character offset in :attr:`text` to one in :attr:`original`."""
        if offset >= len(self._to_original):
            return len(self.original)
        return self._to_original[offset]

    def position(self, offset: int) -> tuple[int, int]:
        """Return the 1-based ``(line, column)`` in the original of a normalized offset."""
        original = self.original_offset(offset)
        lo, hi = 0, len(self._line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._line_starts[mid] <= original:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1, original - self._line_starts[lo] + 1


class _Writer:
    """Accumulates normalized text together with the original offset of each char."""

    def __init__(self) -> None:
        self.chars: list[str] = []
        self.origin: list[int] = []

    def put(self, text: str, origin: int) -> None:
        """Append ``text``; every character maps back to ``origin`` plus its index."""
        for index, char in enumerate(text):
            self.chars.append(char)
            self.origin.append(origin + index)

    def insert(self, text: str, origin: int) -> None:
        """Append text that has no counterpart; all of it maps to ``origin``."""
        for char in text:
            self.chars.append(char)
            self.origin.append(origin)

    @property
    def offset(self) -> int:
        """Offset of the next character to be written."""
        return len(self.chars)


#: Characters that make ``.`` the start of a dotted operator after a number.
_DOT_OPS = frozenset({"^", "*", "/", "\\", "'"})


def _line_col(source: str, offset: int) -> tuple[int, int]:
    """1-based line and column of an offset in ``source``."""
    line = source.count("\n", 0, offset) + 1
    col = offset - (source.rfind("\n", 0, offset) + 1) + 1
    return line, col


def _rest_is_statement_end(source: str, start: int) -> bool:
    """Whether only whitespace then a terminator (newline, ``;``, ``,``, comment) follows."""
    index = start
    while index < len(source) and source[index] in " \t":
        index += 1
    return index >= len(source) or source[index] in "\r\n;,%#"


def _block_marker(line: str) -> str | None:
    """Return ``"{"``/``"}"`` when a line is only a block-comment marker."""
    stripped = line.strip()
    if stripped in ("%{", "#{"):
        return "{"
    if stripped in ("%}", "#}"):
        return "}"
    return None


def normalize(source: str) -> NormalizedSource:
    """Rewrite Octave-only syntax into what the grammar accepts (see module doc).

    :param source: ``.m`` source text.
    :returns: the rewritten text with the map back to ``source``.
    """
    out = _Writer()
    features: list[OctaveFeature] = []
    aug_ops: dict[int, str] = {}
    increments: dict[int, str] = {}

    def note(name: str, offset: int) -> None:
        line, col = _line_col(source, offset)
        features.append(OctaveFeature(name=name, line=line, col=col))

    i = 0
    n = len(source)
    depth = 0  # bracket nesting: separators inside brackets do not end statements
    block_depth = 0  # %{ ... %} nesting
    statement_start = True
    pending_until = False  # an `until` condition is open and needs "; end"
    line_start = True

    while i < n:
        char = source[i]

        # Block comments are recognized per line, before any other lexing.
        if line_start:
            line_end = source.find("\n", i)
            line_end = n if line_end == -1 else line_end
            marker = _block_marker(source[i:line_end])
            if marker == "{" or (block_depth and marker is not None):
                block_depth += 1 if marker == "{" else -1
                text = source[i:line_end]
                if "#" in text:
                    note("#{ #} block comment", i)
                    text = text.replace("#", "%", 1)
                out.put(text, i)
                i = line_end
                line_start = False
                continue
            if block_depth:
                out.put(source[i:line_end], i)
                i = line_end
                line_start = False
                continue
        line_start = False

        if char == "\n":
            if pending_until:
                out.insert("; end", i)
                pending_until = False
            out.put(char, i)
            i += 1
            line_start = True
            if depth == 0:
                statement_start = True
            continue

        if char in " \t\r":
            out.put(char, i)
            i += 1
            continue

        # Comments: `%` or `#` up to the end of the line.
        if char in "%#":
            if pending_until:
                out.insert("; end ", i)
                pending_until = False
            end = source.find("\n", i)
            end = n if end == -1 else end
            if char == "#":
                note("# comment", i)
                out.put("%", i)
                out.put(source[i + 1 : end], i + 1)
            else:
                out.put(source[i:end], i)
            i = end
            continue

        # `...` continues the line; the rest of the line is a comment.
        if source.startswith("...", i):
            end = source.find("\n", i)
            end = n if end == -1 else end
            out.put(source[i:end], i)
            i = end
            continue

        # Strings: double-quoted always; single-quoted unless it is a transpose.
        if char == '"' or (char == "'" and not (i > 0 and source[i - 1] in _TRANSPOSE_AFTER)):
            end = i + 1
            while end < n and source[end] != "\n":
                if source[end] == char:
                    if end + 1 < n and source[end + 1] == char:  # doubled quote
                        end += 2
                        continue
                    end += 1
                    break
                if char == '"' and source[end] == "\\":
                    end += 2
                    continue
                end += 1
            out.put(source[i:end], i)
            i = end
            statement_start = False
            continue

        if char in ";," and depth == 0:
            if pending_until:
                out.insert("; end", i)
                pending_until = False
            out.put(char, i)
            i += 1
            statement_start = True
            continue

        if char in _OPEN:
            depth += 1
        elif char in _CLOSE:
            depth = max(depth - 1, 0)

        # `!=` and `!x`: Octave spellings of `~=` and `~x`.
        if char == "!":
            note("!=" if source.startswith("!=", i) else "! (not)", i)
            out.put("~", i)
            i += 1
            statement_start = False
            continue

        # Prefix increment statement `++x` / `--x`.
        if statement_start and depth == 0 and source.startswith(("++", "--"), i):
            j = i + 2
            while j < n and source[j] in " \t":
                j += 1
            k = j
            while k < n and source[k] in _IDENT_CHARS:
                k += 1
            if j < k and source[j] in _IDENT_START and _rest_is_statement_end(source, k):
                note(source[i : i + 2], i)
                out.put("0+", i)
                out.put(source[i + 2 : j], i + 2)
                increments[out.offset] = "plus" if char == "+" else "minus"
                out.put(source[j:k], j)
                i = k
                statement_start = False
                continue

        if char in _IDENT_START and not (i > 0 and source[i - 1] in _IDENT_CHARS | {"."}):
            k = i
            while k < n and source[k] in _IDENT_CHARS:
                k += 1
            word = source[i:k]
            if word in _END_KEYWORDS:
                note(word, i)
                out.put("end", i)
                out.insert(" " * (len(word) - 3), i + 3)
                i = k
                statement_start = False
                continue
            if (
                word == "do"
                and statement_start
                and depth == 0
                and _rest_is_statement_end(source, k)
            ):
                note("do ... until", i)
                out.insert(f"while {DO_MARKER}", i)
                i = k
                statement_start = False
                continue
            if word == "until" and statement_start and depth == 0:
                note("do ... until", i)
                out.insert(f"{UNTIL_MARKER} =", i)
                i = k
                pending_until = True
                statement_start = False
                continue
            if statement_start and depth == 0:
                # Postfix increment statement `x++` / `x--`.
                j = k
                while j < n and source[j] in " \t":
                    j += 1
                if source.startswith(("++", "--"), j) and _rest_is_statement_end(source, j + 2):
                    note(source[j : j + 2], j)
                    increments[out.offset] = "plus" if source[j] == "+" else "minus"
                    out.put(source[i:j], i)
                    out.put("+0", j)
                    i = j + 2
                    statement_start = False
                    continue
            out.put(word, i)
            i = k
            statement_start = False
            continue

        # Augmented assignment `x op= e` at statement level.
        if (
            depth == 0
            and char in _AUG_OPS
            and source.startswith("=", i + 1)
            and not source.startswith("==", i + 1)
        ):
            note(f"{char}=", i)
            out.put(" ", i)
            aug_ops[out.offset] = _AUG_OPS[char]
            out.put("=", i + 1)
            i += 2
            statement_start = False
            continue

        # `2.^x` is `2 .^ x` in both dialects (the lexer prefers the dotted
        # operator), but the grammar reads the number `2.` and then `^`, a
        # matrix power. A space keeps the dotted operator whole.
        if char.isdigit() and source.startswith(".", i + 1) and source[i + 2 : i + 3] in _DOT_OPS:
            out.put(char, i)
            out.insert(" ", i + 1)
            i += 1
            statement_start = False
            continue

        out.put(char, i)
        i += 1
        statement_start = False

    if pending_until:
        out.insert("; end", n)

    return NormalizedSource(
        original=source,
        text="".join(out.chars),
        _to_original=out.origin,
        features=features,
        aug_ops=aug_ops,
        increments=increments,
    )
