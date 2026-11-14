# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Milestone 6: the MCP server, the promotion check and ``llms.txt``.

The MCP tools are called in-process through the SDK's own ``call_tool``, as a
client would call them, without starting a subprocess. The promotion check
(docs/design.md section 4.3) runs over every rule in both directions: a rule
marked ``mechanical`` that does not earn it fails the suite, and so does a
``guided`` rule that does. ``llms.txt`` must match what the rules generate.
"""

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from conftest import REPO_ROOT, RULES_DIR

from dualect.cli import EXIT_FAILED, EXIT_OK, main
from dualect.llmstxt import render_llms_txt
from dualect.mcp_server import build_server
from dualect.promotion import check_rules, same_m_code
from dualect.rulebook import DIRECTIONS, Status, load_rules

RULES = load_rules(rules_dir=RULES_DIR)
SERVER = build_server(rules_dir=RULES_DIR)


def _call(name: str, **arguments: Any) -> dict[str, Any]:
    result = asyncio.run(SERVER.call_tool(name, arguments))
    assert not getattr(result, "is_error", False), result
    structured = getattr(result, "structured_content", None)
    if structured is None:
        structured = json.loads(result.content[0].text)  # type: ignore[union-attr]
    return dict(structured)


# --- MCP server ------------------------------------------------------------------


def test_mcp_lists_exactly_the_four_tools() -> None:
    tools = asyncio.run(SERVER.list_tools())
    assert sorted(tool.name for tool in tools) == ["get_rule", "scan", "translate", "verify"]


def test_mcp_scan() -> None:
    case = RULES_DIR / "builtins" / "find" / "cases" / "basic.m"
    result = _call("scan", path=str(case))
    assert "builtins/find" in {rule["id"] for rule in result["rules"]}
    assert result["direction"] == "to-python"


def test_mcp_get_rule() -> None:
    result = _call("get_rule", rule_id="indexing/end-in-index", cases=1)
    assert result["status"] == {"to-python": "mechanical", "from-python": "mechanical"}
    assert "x(end)" in result["markdown"] and "```python" in result["markdown"]


def test_mcp_get_rule_unknown_is_an_error() -> None:
    with pytest.raises(Exception, match="no rule"):
        asyncio.run(SERVER.call_tool("get_rule", {"rule_id": "nope/nothing"}))


def test_mcp_translate_without_verifying() -> None:
    case = RULES_DIR / "control-flow" / "while" / "cases" / "basic.py"
    result = _call("translate", path=str(case), verify=False)
    assert result["complete"] and not result["verified"]
    assert result["output"] is None
    assert result["text"].startswith("% Translated from basic.py by dualect. UNVERIFIED")


@pytest.mark.octave
def test_mcp_translate_and_verify(tmp_path: Path) -> None:
    case = RULES_DIR / "indexing" / "ranges" / "cases" / "basic.m"
    output = tmp_path / "ranges.py"
    result = _call("translate", path=str(case), output=str(output))
    assert result["verified"], result["report"]
    assert output.read_text(encoding="utf-8") == result["text"]
    checked = _call("verify", m_file=str(case), py_file=str(output))
    assert checked["ok"], checked["report"]


# --- promotion check -----------------------------------------------------------------


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_declared_mechanical_rules_qualify_without_execution(direction: str) -> None:
    results = check_rules(rules=RULES, directions=(direction,), verify=False)
    problems = [r.line() for r in results if r.declared is Status.MECHANICAL and r.problem]
    assert problems == []


@pytest.mark.octave
def test_every_rule_earns_its_status_in_both_directions() -> None:
    results = check_rules(rules=RULES, verify=True)
    problems = [r.line() for r in results if r.problem]
    assert problems == []


def test_check_fails_for_a_mechanical_rule_that_does_not_qualify(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rules_dir = tmp_path / "rules"
    shutil.copytree(RULES_DIR, rules_dir, ignore=shutil.ignore_patterns("__pycache__"))
    case = rules_dir / "operators" / "power" / "cases" / "basic.py"
    case.write_text(case.read_text(encoding="utf-8").replace("Q = A**2", "Q = A @ A"))
    code = main(
        ["rules", "--rules-dir", str(rules_dir), "check", "operators/power",
         "--direction", "to-python", "--no-verify"]
    )  # fmt: skip
    assert code == EXIT_FAILED
    assert "marked mechanical but does not qualify" in capsys.readouterr().out


def test_check_command_passes_on_the_repository(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["rules", "check", "indexing/ranges", "--no-verify"]) == EXIT_OK
    assert "0 with a status problem" in capsys.readouterr().out


def test_m_layer2_compares_structure_not_layout() -> None:
    assert same_m_code("x = [1 2 3];\ny = x(end);\n", "x = [1, 2, 3];  % note\ny = x(end);\n")
    assert not same_m_code("y = x(1);\n", "y = x(2);\n")


# --- llms.txt ----------------------------------------------------------------------


def test_llms_txt_is_current() -> None:
    committed = (REPO_ROOT / "llms.txt").read_text(encoding="utf-8")
    assert committed == render_llms_txt(rules=RULES), "run: uv run dualect llms-txt -o llms.txt"


def test_llms_txt_has_one_line_per_rule() -> None:
    text = render_llms_txt(rules=RULES)
    lines = [line for line in text.splitlines() if line.startswith("- [")]
    assert len(lines) == len(RULES)
    assert all(f"[{rule.id}]" in text for rule in RULES)


def test_llms_txt_command(tmp_path: Path) -> None:
    output = tmp_path / "llms.txt"
    assert main(["llms-txt", "-o", str(output)]) == EXIT_OK
    assert output.read_text(encoding="utf-8") == render_llms_txt(rules=RULES)
