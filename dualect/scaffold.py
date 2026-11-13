# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""``dualect rules new``: create a rule folder from a template.

The new rule starts as ``unsupported`` with ``TODO`` placeholders. It passes the
schema (so tooling can load it straight away), but the test suite rejects any
rule that still contains ``TODO``, so a half-written rule cannot be merged.
"""

import re
from pathlib import Path

from dualect.errors import RuleError
from dualect.rulebook import CASES_DIR, RULE_FILE

#: Marker the test suite searches for; anything carrying it is unfinished.
PLACEHOLDER = "TODO"

_RULE_ID = re.compile(r"^[a-z][a-z0-9-]*/[a-z][a-z0-9-]*$")

_RULE_TEMPLATE = """\
id: {rule_id}
title: "{PLACEHOLDER}: short title for {rule_id}"
status: unsupported           # unsupported | guided | mechanical
dialects: [octave, matlab]
octave: |
  % {PLACEHOLDER}: the .m form
python: |
  # {PLACEHOLDER}: the idiomatic NumPy form
why: >
  {PLACEHOLDER}: why the two forms differ, in two or three sentences.
pitfalls: []
related: []
python_imports: [import numpy as np]
"""

_CASE_M = """\
% {PLACEHOLDER}: a short .m script; every variable it leaves behind is compared.
% Directives: '% dualect: index <names>' marks 1-based index outputs,
% '% dualect: ignore <names>' skips variables that legitimately differ.
"""

_CASE_PY = """\
# {PLACEHOLDER}: the hand-written idiomatic NumPy twin of basic.m.
import numpy as np
"""


def new_rule(*, rules_dir: Path, rule_id: str) -> Path:
    """Create ``rules_dir/<category>/<rule-id>/`` with a template rule and case pair.

    :param rules_dir: root of the knowledge base.
    :param rule_id: ``<category>/<rule-id>``, lowercase with hyphens.
    :returns: the new rule folder.
    :raises RuleError: if the id is malformed or the folder already exists.
    """
    if not _RULE_ID.match(rule_id):
        raise RuleError(
            f"invalid rule id {rule_id!r}: expected <category>/<rule-id>, "
            "lowercase letters, digits and hyphens"
        )
    rule_dir = rules_dir / rule_id
    if rule_dir.exists():
        raise RuleError(f"{rule_dir} already exists")

    cases_dir = rule_dir / CASES_DIR
    cases_dir.mkdir(parents=True)
    (rule_dir / RULE_FILE).write_text(
        _RULE_TEMPLATE.format(rule_id=rule_id, PLACEHOLDER=PLACEHOLDER), encoding="utf-8"
    )
    (cases_dir / "basic.m").write_text(_CASE_M.format(PLACEHOLDER=PLACEHOLDER), encoding="utf-8")
    (cases_dir / "basic.py").write_text(_CASE_PY.format(PLACEHOLDER=PLACEHOLDER), encoding="utf-8")
    return rule_dir
