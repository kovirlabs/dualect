# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Type-check every ``rules/**/translate.py`` with the repository's mypy settings.

The translators are loaded at run time under the module names
``dualect_rules.<category>.<rule_id>.translate`` (hyphens become underscores,
see :func:`dualect.translators.load_translator`). Their folders are not valid
package names and every file is called ``translate.py``, so ``mypy rules``
cannot tell them apart. This script links each file into a temporary package
tree under its run-time name and checks that tree in one mypy run.

Usage: ``uv run python tools/mypy_rules.py``.
"""

import sys
import tempfile
from pathlib import Path

from mypy import api

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    """Run mypy over the translators; return its exit status."""
    files = sorted((REPO / "rules").glob("*/*/translate.py"))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "dualect_rules"
        targets = []
        for path in files:
            category, rule = path.parent.parent.name, path.parent.name
            package = root / category.replace("-", "_") / rule.replace("-", "_")
            package.mkdir(parents=True, exist_ok=True)
            for folder in (root, package.parent, package):
                (folder / "__init__.py").touch()
            # A copy, not a link: mypy reports paths, and a copy keeps them in
            # the temporary tree; errors are mapped back below.
            target = package / "translate.py"
            target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            targets.append((target, path))
        stdout, stderr, status = api.run(
            [
                "--config-file",
                str(REPO / "pyproject.toml"),
                "--explicit-package-bases",
                *(str(t) for t, _ in targets),
            ]
        )
        for target, path in targets:
            stdout = stdout.replace(str(target), str(path.relative_to(REPO)))
    sys.stdout.write(stdout.replace(f"{tmp}/", ""))
    sys.stderr.write(stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
