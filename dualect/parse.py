# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Parsing ``.m`` source with tree-sitter (docs/design.md section 9).

The grammar is ``tree-sitter-matlab``; Octave-only syntax is rewritten first by
:mod:`dualect.source`. Locations reported anywhere in dualect are positions in
the *original* text, mapped back through the normalization.
"""

import functools
from dataclasses import dataclass, field

import tree_sitter_matlab
from tree_sitter import Language, Node, Parser, Tree

from dualect.ir import Loc
from dualect.source import NormalizedSource, normalize


@functools.cache
def _parser() -> Parser:
    """Build the parser once; loading the grammar is the expensive part."""
    return Parser(Language(tree_sitter_matlab.language()))


@dataclass
class ParsedSource:
    """A parsed ``.m`` file.

    :param source: the normalized source and its map to the original.
    :param tree: the tree-sitter syntax tree of the normalized text.
    """

    source: NormalizedSource
    tree: Tree
    _byte_to_char: list[int] = field(repr=False)

    def char_offset(self, byte_offset: int) -> int:
        """Convert a tree-sitter byte offset into a normalized character offset."""
        if byte_offset >= len(self._byte_to_char):
            return len(self.source.text)
        return self._byte_to_char[byte_offset]

    def text(self, node: Node) -> str:
        """The normalized text a node spans."""
        return self.source.text[self.char_offset(node.start_byte) : self.char_offset(node.end_byte)]

    def original_text(self, node: Node) -> str:
        """The original source text a node spans (what the user wrote)."""
        start = self.source.original_offset(self.char_offset(node.start_byte))
        end = self.source.original_offset(self.char_offset(node.end_byte))
        return self.source.original[start:end]

    def loc(self, node: Node) -> Loc:
        """The node's span in the original source."""
        line, col = self.source.position(self.char_offset(node.start_byte))
        end_line, end_col = self.source.position(self.char_offset(node.end_byte))
        return Loc(line, col, end_line, end_col)


def parse_m(text: str) -> ParsedSource:
    """Parse ``.m`` source text.

    Parsing never raises: regions the grammar cannot read become ``ERROR`` (or
    zero-width ``MISSING``) nodes, which :mod:`dualect.lower` turns into named
    ``parse:error`` constructs so they are reported, not dropped.

    :param text: the source.
    """
    normalized = normalize(text)
    encoded = normalized.text.encode("utf-8")
    byte_to_char: list[int] = []
    for index, char in enumerate(normalized.text):
        byte_to_char.extend([index] * len(char.encode("utf-8")))
    tree = _parser().parse(encoded)
    return ParsedSource(source=normalized, tree=tree, _byte_to_char=byte_to_char)
