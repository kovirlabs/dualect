# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""``dualect translate``: the whole pipeline of docs/design.md section 5.

parse -> scan -> plan -> translate -> emit -> verify -> report. Planning and
translation happen statement by statement in :mod:`dualect.emit`; this module
assembles the file (header, imports, ruff formatting), writes it, verifies it
against the source by execution, and reports every statement with the rules
it needed, the status used and whether the result was verified.

The output always says what it is. Its first line states whether the file was
verified; untranslated statements are holes that raise ``NotImplementedError``;
an unverified file is labelled UNVERIFIED with the reason.
"""

from dataclasses import dataclass, field
from pathlib import Path

from dualect.compare import Tolerance
from dualect.dialect import Dialect
from dualect.emit import Context, StmtReport, render
from dualect.errors import DualectError, ExecutionError, RuntimeMissingError
from dualect.pyformat import format_python
from dualect.rulebook import Rule, Status
from dualect.runners import MRunner
from dualect.scan import RuleIndex, ScanResult, scan_text
from dualect.translators import TranslatorSet
from dualect.verify import VerifyResult, options_for_pair, verify_pair

UNVERIFIED = "UNVERIFIED"


class StrictModeError(DualectError):
    """``--strict``: the file needs guided or unsupported translation."""


@dataclass
class Translation:
    """The result of translating one file.

    :param source_name: the ``.m`` file.
    :param code: the Python source (formatted), without the status header.
    :param statements: one report per statement, in source order.
    :param scan: the scan the translation was planned from.
    :param verify: the verification result, when verification ran.
    :param verify_note: why verification did not run, or how it failed.
    """

    source_name: str
    code: str
    statements: list[StmtReport]
    scan: ScanResult
    verify: VerifyResult | None = None
    verify_note: str = ""
    rules_used: list[Rule] = field(default_factory=list)

    @property
    def holes(self) -> list[StmtReport]:
        """Statements that were not translated mechanically."""
        return [s for s in self.statements if s.status is not Status.MECHANICAL]

    @property
    def complete(self) -> bool:
        """Every statement was translated mechanically."""
        return not self.holes

    @property
    def verified(self) -> bool:
        """The output ran and matched the source on every compared variable."""
        return self.verify is not None and self.verify.ok

    def header(self) -> str:
        """The first line of the output file: what this file is and whether it was checked."""
        name = Path(self.source_name).name
        if self.verified:
            return f"# Translated from {name} by dualect; verified against the .m source."
        reason = self.verify_note or "not verified"
        return f"# Translated from {name} by dualect. {UNVERIFIED}: {reason}."

    def text(self) -> str:
        """The complete output file."""
        return f"{self.header()}\n{self.code}"

    def report(self) -> str:
        """Per-statement report: location, status used, verified, rules."""
        verified = "yes" if self.verified else "no"
        lines = [f"translate {self.source_name}"]
        for stmt in self.statements:
            rules = ", ".join(stmt.rules) or "-"
            status = stmt.status.value
            lines.append(
                f"  {str(stmt.loc):<7} {stmt.kind:<10} {status:<11} {verified:<3}  {rules}"
            )
            if stmt.reason:
                lines.append(f"          {stmt.reason}")
        counts = {status: 0 for status in Status}
        for stmt in self.statements:
            counts[stmt.status] += 1
        lines.append(
            f"{len(self.statements)} statements: {counts[Status.MECHANICAL]} mechanical, "
            f"{counts[Status.GUIDED]} guided, {counts[Status.UNSUPPORTED]} unsupported"
        )
        if self.verify is not None:
            lines.append(self.verify.report())
        if self.verified:
            lines.append("verified: yes")
        else:
            lines.append(f"verified: no ({self.verify_note or 'not run'})")
        if not self.complete:
            lines.append(
                "untranslated statements are marked TODO(dualect) in the output; "
                f"`dualect context {self.source_name}` prints the rules an LLM needs for them"
            )
        return "\n".join(lines)


def translate_scan(
    *, scan: ScanResult, index: RuleIndex, translators: TranslatorSet
) -> Translation:
    """Translate a scanned file into formatted Python (no verification).

    :param scan: the scanner's result for the file.
    :param index: the rules, by construct tag.
    :param translators: the rules' translator modules.
    """
    ctx = Context(
        scan=scan,
        translator_for=translators.get,
        rule_for=index.rule_for,
        rule_by_id=index.rule_by_id,
    )
    ctx.enter_script()
    lines = ctx.statements(scan.program.body)
    body = render(lines)
    imports = "".join(f"{line}\n" for line in sorted(ctx.imports))
    code = format_python(f"{imports}\n{body}" if imports else body)
    return Translation(
        source_name=scan.source_name,
        code=code,
        statements=ctx.reports,
        scan=scan,
        rules_used=scan.needed_rules(),
    )


def translate_text(
    *, text: str, name: str, rules: list[Rule], dialect: Dialect, strict: bool = False
) -> Translation:
    """Scan and translate ``.m`` source text (no verification).

    :param text: the source.
    :param name: its file name, for reports.
    :param rules: all loaded rules.
    :param dialect: the source dialect.
    :param strict: raise :class:`StrictModeError` instead of leaving holes.
    """
    index = RuleIndex(rules=rules, dialect=dialect)
    scan = scan_text(text=text, name=name, index=index)
    result = translate_scan(scan=scan, index=index, translators=TranslatorSet(rules=rules))
    if strict and not result.complete:
        raise StrictModeError(
            f"--strict: {len(result.holes)} statement(s) need guided or unsupported "
            f"translation\n{result.report()}"
        )
    return result


def tolerance_for(rules: list[Rule]) -> Tolerance:
    """The loosest per-rule tolerance among the rules used (or the default)."""
    tolerance = Tolerance()
    rtol, atol = tolerance.rtol, tolerance.atol
    for rule in rules:
        rtol = max(rtol, rule.tolerance.get("rtol", rtol))
        atol = max(atol, rule.tolerance.get("atol", atol))
    return Tolerance(rtol=rtol, atol=atol)


def verify_translation(
    *, result: Translation, source: Path, output: Path, runner: MRunner | None = None
) -> None:
    """Run the source and the written output and compare; updates ``result``.

    :param result: the translation (already written to ``output``).
    :param source: the ``.m`` file.
    :param output: the ``.py`` file.
    :param runner: an Octave session to reuse.
    """
    if not result.complete:
        result.verify_note = f"{len(result.holes)} statement(s) not translated"
        return
    options = options_for_pair(
        m_path=source, py_path=output, tolerance=tolerance_for(result.rules_used)
    )
    try:
        result.verify = verify_pair(m_path=source, py_path=output, options=options, runner=runner)
    except RuntimeMissingError as exc:
        result.verify_note = f"cannot run the .m side ({exc})"
        return
    except ExecutionError as exc:
        result.verify_note = f"execution failed: {exc}"
        return
    if not result.verify.ok:
        result.verify_note = "outputs differ from the .m source"


def translate_file(
    *,
    source: Path,
    output: Path,
    rules: list[Rule],
    dialect: Dialect,
    strict: bool = False,
    verify: bool = True,
    runner: MRunner | None = None,
) -> Translation:
    """The full pipeline for one file: translate, write, verify, rewrite the header.

    :param source: the ``.m`` file.
    :param output: where to write the ``.py``.
    :param rules: all loaded rules.
    :param dialect: the source dialect.
    :param strict: stop (raise, write nothing) unless every statement is mechanical.
    :param verify: run both sides and compare (needs Octave).
    :param runner: an Octave session to reuse.
    """
    text = source.read_text(encoding="utf-8")
    result = translate_text(
        text=text, name=str(source), rules=rules, dialect=dialect, strict=strict
    )
    if not verify:
        result.verify_note = "verification was not requested"
    output.write_text(result.text(), encoding="utf-8")
    if verify:
        verify_translation(result=result, source=source, output=output, runner=runner)
        output.write_text(result.text(), encoding="utf-8")
    return result
