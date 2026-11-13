# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""``dualect scan``: which rules does a file need? (docs/design.md section 6).

The scanner parses and lowers the file, infers shapes, tags every statement's
constructs (:mod:`dualect.constructs`) and looks each tag up in the rules'
``match`` lists. A tag no rule matches is reported as unsupported under its
own name; so is text the parser could not read, and (under ``--dialect
matlab``) Octave-only syntax.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dualect import ir
from dualect.constructs import Use, statement_uses
from dualect.dialect import Dialect
from dualect.errors import RuleError
from dualect.infer import Analysis, analyze
from dualect.lower import lower_parsed
from dualect.parse import parse_m
from dualect.rulebook import Rule, Status
from dualect.source import OctaveFeature

#: Tag for Octave-only syntax found while scanning for the matlab dialect.
OCTAVE_ONLY_TAG = "dialect:octave-only"


class RuleIndex:
    """Maps construct tags to the rule that covers them, for one dialect."""

    def __init__(self, *, rules: Iterable[Rule], dialect: Dialect) -> None:
        """Index the rules that apply to ``dialect``.

        :param rules: all loaded rules.
        :param dialect: the source dialect; rules for other dialects or
            versions are left out.
        :raises RuleError: when two applicable rules claim the same tag.
        """
        self.dialect = dialect
        self.by_tag: dict[str, Rule] = {}
        for rule in rules:
            if not dialect.allows(rule.data):
                continue
            for tag in rule.match:
                other = self.by_tag.get(tag)
                if other is not None:
                    raise RuleError(
                        f"{rule.id} and {other.id} both match {tag!r} for dialect {dialect}"
                    )
                self.by_tag[tag] = rule

    def rule_for(self, tag: str) -> Rule | None:
        """The rule covering ``tag``, or ``None``."""
        return self.by_tag.get(tag)


@dataclass(frozen=True)
class Finding:
    """One construct use, resolved to a rule.

    :param tag: construct tag.
    :param loc: where it is.
    :param rule: the rule covering it, or ``None`` when no rule does.
    :param text: a short source excerpt, for reports.
    """

    tag: str
    loc: ir.Loc
    rule: Rule | None
    text: str = ""

    @property
    def status(self) -> Status:
        """The rule's status; ``unsupported`` when there is no rule."""
        return self.rule.status if self.rule is not None else Status.UNSUPPORTED

    @property
    def unsupported(self) -> bool:
        """Whether this use has no rule or only an ``unsupported`` one."""
        return self.status is Status.UNSUPPORTED


@dataclass
class ScanResult:
    """Everything the scanner learned about one file.

    :param source_name: the file, for reports.
    :param program: the lowered IR.
    :param analysis: inference results.
    :param findings: every construct use, in source order.
    :param uses_by_stmt: ``id(stmt)`` -> the findings of that statement itself.
    :param features: Octave-only syntax found by the normalizer.
    """

    source_name: str
    program: ir.Program
    analysis: Analysis
    findings: list[Finding] = field(default_factory=list)
    uses_by_stmt: dict[int, list[Finding]] = field(default_factory=dict)
    features: list[OctaveFeature] = field(default_factory=list)

    def needed_rules(self) -> list[Rule]:
        """The rules this file needs, sorted by id."""
        rules = {f.rule.id: f.rule for f in self.findings if f.rule is not None}
        return [rules[key] for key in sorted(rules)]

    def unsupported(self) -> list[Finding]:
        """Uses with no rule (or an ``unsupported`` rule), in source order."""
        return [f for f in self.findings if f.unsupported]

    def report(self) -> str:
        """Human-readable listing: each rule with its locations, then the gaps."""
        lines = [f"scan {self.source_name}"]
        rules = self.needed_rules()
        width = max((len(rule.id) for rule in rules), default=0)
        for rule in rules:
            locs = [str(f.loc) for f in self.findings if f.rule is rule]
            shown = ", ".join(_unique(locs)[:8])
            more = len(_unique(locs)) - 8
            suffix = f" (+{more} more)" if more > 0 else ""
            lines.append(f"  {rule.id:<{width}}  {rule.status.value:<11} {shown}{suffix}")
        gaps = self.unsupported()
        if gaps:
            lines.append(f"unsupported ({len(gaps)}):")
            for finding in gaps:
                label = finding.rule.id if finding.rule is not None else "no rule"
                excerpt = f"  {finding.text}" if finding.text else ""
                lines.append(f"  {str(finding.loc):<7} {finding.tag}  ({label}){excerpt}")
        counts = {status: 0 for status in Status}
        for rule in rules:
            counts[rule.status] += 1
        lines.append(
            f"{len(rules)} rules: {counts[Status.MECHANICAL]} mechanical, "
            f"{counts[Status.GUIDED]} guided, {counts[Status.UNSUPPORTED]} unsupported; "
            f"{len(gaps)} unsupported construct uses"
        )
        return "\n".join(lines)

    def as_json(self) -> dict[str, Any]:
        """Machine-readable form (for tools and, later, the MCP server)."""
        return {
            "source": self.source_name,
            "rules": [
                {
                    "id": rule.id,
                    "status": rule.status.value,
                    "locations": _unique(str(f.loc) for f in self.findings if f.rule is rule),
                }
                for rule in self.needed_rules()
            ],
            "unsupported": [
                {
                    "tag": f.tag,
                    "location": str(f.loc),
                    "rule": f.rule.id if f.rule is not None else None,
                    "text": f.text,
                }
                for f in self.unsupported()
            ],
        }

    def to_json(self) -> str:
        """:meth:`as_json` serialized."""
        return json.dumps(self.as_json(), indent=2)


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _excerpt(text: str, loc: ir.Loc) -> str:
    """The first source line of a span, trimmed, for reports."""
    lines = text.splitlines()
    if not 1 <= loc.line <= len(lines):
        return ""
    line = lines[loc.line - 1]
    start = max(loc.col - 1, 0)
    end = loc.end_col - 1 if loc.end_line == loc.line else len(line)
    snippet = line[start:end].strip() or line.strip()
    return snippet if len(snippet) <= 48 else snippet[:45] + "..."


def scan_text(*, text: str, name: str, index: RuleIndex) -> ScanResult:
    """Scan ``.m`` source text.

    :param text: the source.
    :param name: file name for reports.
    :param index: the rules to match against (already filtered by dialect).
    """
    parsed = parse_m(text)
    program = lower_parsed(parsed=parsed, name=name, dialect=index.dialect.name)
    analysis = analyze(program)
    result = ScanResult(
        source_name=name, program=program, analysis=analysis, features=parsed.source.features
    )
    for stmt in ir.walk_stmts(program.body):
        found = [
            Finding(
                tag=use.tag,
                loc=use.loc,
                rule=index.rule_for(use.tag),
                text=_excerpt(text, use.loc) if index.rule_for(use.tag) is None else "",
            )
            for use in statement_uses(stmt, analysis)
        ]
        result.uses_by_stmt[id(stmt)] = found
        result.findings.extend(found)
    if index.dialect.name == "matlab":
        for feature in parsed.source.features:
            loc = ir.Loc(feature.line, feature.col, feature.line, feature.col + len(feature.name))
            finding = Finding(
                tag=OCTAVE_ONLY_TAG, loc=loc, rule=None, text=f"{feature.name} is Octave-only"
            )
            result.findings.append(finding)
    result.findings.sort(key=lambda f: (f.loc.line, f.loc.col))
    return result


def scan_file(*, path: Path, rules: list[Rule], dialect: Dialect) -> ScanResult:
    """Scan a ``.m`` file.

    :param path: the file.
    :param rules: all loaded rules.
    :param dialect: the source dialect.
    """
    text = path.read_text(encoding="utf-8")
    return scan_text(text=text, name=str(path), index=RuleIndex(rules=rules, dialect=dialect))


def uses_of(findings: Iterable[Finding]) -> list[Use]:
    """Strip findings back to plain uses (tag and location)."""
    return [Use(f.tag, f.loc) for f in findings]
