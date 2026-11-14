# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""The MCP server (docs/design.md section 6): dualect's tools for an LLM client.

``dualect mcp`` serves four tools over stdio, built on the official ``mcp``
Python SDK:

* ``scan``: the rules a ``.m`` or ``.py`` file needs, and the constructs that
  have none (the same data as ``dualect scan --json``);
* ``get_rule``: one rule's ``rule.yaml`` and cases, as LLM-ready context;
* ``translate``: translate a file in either direction, verify it by running
  both sides, and return the output with its per-statement report;
* ``verify``: run a ``.m`` and a ``.py`` file and compare their variables.

The server never calls an LLM API (section 12): it gives the client's model the
knowledge and the checks, and whatever the model writes comes back through
``verify`` like everything else.
"""

import tempfile
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from dualect import __version__
from dualect.compare import Tolerance
from dualect.context import render_rule
from dualect.dialect import parse_dialect
from dualect.errors import DualectError
from dualect.paths import default_rules_dir
from dualect.pipeline import StrictModeError, translate_file
from dualect.rulebook import FROM_PYTHON, TO_PYTHON, Rule, load_rules
from dualect.scan import scan_file
from dualect.verify import options_for_pair, verify_pair

INSTRUCTIONS = """\
dualect translates numeric code between .m files (Octave/MATLAB style) and
Python/NumPy, one tested rule per construct, and verifies every translation by
running both sides. Use `scan` to see which rules a file needs, `get_rule` to
read a rule (why the languages differ, pitfalls, verified example pairs),
`translate` for the deterministic translation (statements it cannot translate
are left as TODO(dualect) holes and reported), and `verify` to check any
translation, including your own, by execution. Never present an unverified
translation as verified."""


def _source(path: str) -> Path:
    source = Path(path).expanduser()
    if not source.is_file():
        raise ToolError(f"no such file: {path}")
    if source.suffix not in (".m", ".py"):
        raise ToolError(f"expected a .m or .py file, got {path}")
    return source


def build_server(*, rules_dir: Path | None = None) -> MCPServer:
    """Create the server with its four tools.

    :param rules_dir: the rules folder (default: the repository's or the
        installed copy, see :func:`dualect.paths.default_rules_dir`).
    """
    root = rules_dir or default_rules_dir()
    server = MCPServer(name="dualect", version=__version__, instructions=INSTRUCTIONS)

    def rules() -> list[Rule]:
        return load_rules(rules_dir=root)

    @server.tool()
    def scan(path: str, dialect: str | None = None) -> dict[str, Any]:
        """List the rules a .m or .py file needs and the constructs no rule covers.

        :param path: the source file (.m: translated to Python; .py: to .m).
        :param dialect: for a .m file, the source dialect, e.g. "octave" or "matlab@R2023a".
        """
        try:
            result = scan_file(path=_source(path), rules=rules(), dialect=parse_dialect(dialect))
        except DualectError as exc:
            raise ToolError(str(exc)) from exc
        return result.as_json()

    @server.tool()
    def get_rule(rule_id: str, cases: int = 2) -> dict[str, Any]:
        """Return one rule: its rule.yaml (the knowledge) and up to `cases` verified example pairs.

        :param rule_id: e.g. "indexing/end-in-index" (see `scan` for the ids a file needs).
        :param cases: how many example pairs to include.
        """
        found = {rule.id: rule for rule in rules()}
        rule = found.get(rule_id)
        if rule is None:
            raise ToolError(f"no rule {rule_id!r}; known: {', '.join(sorted(found))}")
        return {
            "id": rule.id,
            "title": rule.title,
            "status": {
                TO_PYTHON: rule.status.value,
                FROM_PYTHON: rule.python_status.value if rule.python_status else None,
            },
            "markdown": render_rule(rule, limit=max(cases, 0)),
        }

    @server.tool()
    def translate(
        path: str,
        output: str | None = None,
        dialect: str | None = None,
        strict: bool = False,
        verify: bool = True,
    ) -> dict[str, Any]:
        """Translate a .m file to Python or a .py file to .m, then verify it by running both.

        :param path: the source file; its suffix picks the direction.
        :param output: where to write the result; by default nothing is written
            and the text is returned.
        :param dialect: for a .m file, the source dialect.
        :param strict: fail instead of leaving holes for untranslated statements.
        :param verify: run both sides and compare (needs GNU Octave).
        """
        source = _source(path)
        suffix = ".m" if source.suffix == ".py" else ".py"
        with tempfile.TemporaryDirectory() as scratch:
            target = (
                Path(output).expanduser() if output else Path(scratch) / f"{source.stem}{suffix}"
            )
            try:
                result = translate_file(
                    source=source,
                    output=target,
                    rules=rules(),
                    dialect=parse_dialect(dialect),
                    strict=strict,
                    verify=verify,
                )
            except StrictModeError as exc:
                raise ToolError(f"{exc}\nnothing was written") from exc
            except DualectError as exc:
                raise ToolError(str(exc)) from exc
            text = target.read_text(encoding="utf-8")
        return {
            "output": str(target) if output else None,
            "text": text,
            "verified": result.verified,
            "complete": result.complete,
            "holes": [
                {"line": h.loc.line, "status": h.status.value, "reason": h.reason}
                for h in result.holes
            ],
            "report": result.report(),
        }

    @server.tool()
    def verify(m_file: str, py_file: str) -> dict[str, Any]:
        """Run a .m script and a .py script and compare every variable the .m script leaves.

        :param m_file: the .m script (run in GNU Octave).
        :param py_file: the Python script.
        """
        m_path, py_path = _source(m_file), _source(py_file)
        try:
            options = options_for_pair(m_path=m_path, py_path=py_path, tolerance=Tolerance())
            result = verify_pair(m_path=m_path, py_path=py_path, options=options)
        except DualectError as exc:
            raise ToolError(str(exc)) from exc
        return {
            "ok": result.ok,
            "variables": [
                {"name": v.name, "ok": v.ok, "detail": v.detail} for v in result.variables
            ],
            "python_only": list(result.python_only),
            "report": result.report(),
        }

    return server


def serve(*, rules_dir: Path | None = None) -> None:
    """Run the server on stdio until the client disconnects.

    :param rules_dir: the rules folder (default: see :func:`build_server`).
    """
    build_server(rules_dir=rules_dir).run("stdio")
