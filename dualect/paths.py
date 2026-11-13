# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Where dualect finds its rule knowledge base."""

import os
from pathlib import Path

#: Environment variable that points dualect at a different ``rules/`` folder,
#: e.g. a working copy while editing rules outside the repository checkout.
RULES_DIR_ENV = "DUALECT_RULES_DIR"

_PACKAGE_DIR = Path(__file__).resolve().parent


def default_rules_dir() -> Path:
    """Return the rules folder dualect should use when none is given.

    Order: the ``DUALECT_RULES_DIR`` environment variable; the repository's
    ``rules/`` next to the package (a source checkout or editable install); the
    copy bundled into a built wheel as ``dualect/_rules``.

    :returns: path to the folder holding ``<category>/<rule-id>/`` subfolders.
    """
    override = os.environ.get(RULES_DIR_ENV)
    if override:
        return Path(override)
    source_tree = _PACKAGE_DIR.parent / "rules"
    if source_tree.is_dir():
        return source_tree
    return _PACKAGE_DIR / "_rules"


def schema_path() -> Path:
    """Return the path of the JSON Schema that every ``rule.yaml`` must satisfy."""
    return _PACKAGE_DIR / "schema" / "rule.schema.json"
