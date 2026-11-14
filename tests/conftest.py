# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Shared pytest setup.

Tests that execute .m code carry ``@pytest.mark.octave``. They are skipped when
Octave is not installed, so the rest of the suite runs anywhere. CI sets
``DUALECT_REQUIRE_OCTAVE=1`` so a missing Octave fails loudly there instead of
turning every execution test into a silent skip.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from dualect.runners import MRunner, m_runtime_available

REQUIRE_ENV = "DUALECT_REQUIRE_OCTAVE"
REPO_ROOT = Path(__file__).resolve().parent.parent
RULES_DIR = REPO_ROOT / "rules"
DATA_DIR = Path(__file__).resolve().parent / "data"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip Octave tests when Octave is absent, unless the environment demands it."""
    if m_runtime_available() or os.environ.get(REQUIRE_ENV):
        return
    skip = pytest.mark.skip(reason="GNU Octave is not installed")
    for item in items:
        if "octave" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def m_runner() -> Iterator[MRunner]:
    """One Octave session for the whole run; starting Octave costs about a second."""
    with MRunner() as runner:
        yield runner
