# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""``dualect context``: the rules a file needs, as LLM-ready prompt context.

The repository is the LLM's memory (docs/design.md section 6): instead of
re-deriving the translation rules, a model is handed exactly the rules one file
needs, each as its ``rule.yaml`` plus one or two paired cases. dualect never
calls a model itself; whatever model the user runs reads this text, and its
translation comes back through ``dualect verify``.
"""

from dualect.rulebook import TO_PYTHON, CasePair, Rule, Status
from dualect.scan import ScanResult

#: Cases shown per rule: enough to show the pattern, small enough to stay focused.
MAX_CASES = 2


def pick_cases(rule: Rule, *, limit: int = MAX_CASES) -> list[CasePair]:
    """Choose the cases to show for a rule: ``basic`` first, then by name.

    :param rule: the rule.
    :param limit: how many at most.
    """
    ordered = sorted(rule.cases, key=lambda case: (case.name != "basic", case.name))
    return ordered[:limit]


def render_rule(rule: Rule, *, limit: int = MAX_CASES, direction: str = TO_PYTHON) -> str:
    """One rule as a Markdown section: its ``rule.yaml`` and its chosen cases.

    :param rule: the rule.
    :param limit: how many cases to include.
    :param direction: which status to show in the heading.
    """
    status = rule.status_for(direction) or Status.UNSUPPORTED
    parts = [
        f"## {rule.id} ({status.value} {direction})",
        "",
        "```yaml",
        (rule.path / "rule.yaml").read_text(encoding="utf-8").rstrip(),
        "```",
    ]
    for case in pick_cases(rule, limit=limit):
        parts += [
            "",
            f"Case `{case.name}`, .m and its verified Python twin:",
            "",
            "```octave",
            case.m_path.read_text(encoding="utf-8").rstrip(),
            "```",
            "",
            "```python",
            case.py_path.read_text(encoding="utf-8").rstrip(),
            "```",
        ]
    return "\n".join(parts)


def build_context(*, result: ScanResult, limit: int = MAX_CASES) -> str:
    """The prompt context for one scanned file.

    A two-line header names the file, the rules and the constructs that have no
    rule; then each needed rule follows. Nothing else is included.

    :param result: the scan of the file.
    :param limit: cases per rule.
    """
    rules = result.needed_rules()
    header = [
        f"# dualect rules for {result.source_name}",
        "",
        f"Rules this file needs: {', '.join(rule.id for rule in rules) or 'none'}.",
    ]
    gaps = result.unsupported()
    if gaps:
        named = ", ".join(f"{gap.tag} at {gap.loc}" for gap in gaps)
        header.append(f"Constructs with no rule (translate with care and say so): {named}.")
    sections = [render_rule(rule, limit=limit, direction=result.direction) for rule in rules]
    return "\n\n".join(["\n".join(header), *sections]) + "\n"
