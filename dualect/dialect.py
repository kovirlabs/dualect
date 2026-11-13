# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Source dialects and versions (docs/design.md section 4.4).

A dialect spec is ``octave`` or ``matlab``, optionally with a version:
``octave@8.4``, ``matlab@R2023a``. Rules are filtered by their ``dialects``
list and their ``since`` (inclusive) / ``until`` (exclusive) bounds, so a
behavior change can live in a new variant rule without editing the old one.
"""

import re
from dataclasses import dataclass
from typing import Any

from dualect.errors import DualectError

DIALECTS = ("octave", "matlab")
DEFAULT_DIALECT = "octave"

_OCTAVE_VERSION = re.compile(r"^[0-9]+(\.[0-9]+)*$")
_MATLAB_VERSION = re.compile(r"^R([0-9]{4})([ab])$")


class DialectError(DualectError):
    """A ``--dialect`` value could not be understood."""


@dataclass(frozen=True)
class Dialect:
    """A source dialect, optionally pinned to a version.

    :param name: ``"octave"`` or ``"matlab"``.
    :param version: ``"8.4"`` / ``"R2023a"``, or ``None`` for "any version".
    """

    name: str = DEFAULT_DIALECT
    version: str | None = None

    def __str__(self) -> str:
        """Render as ``name`` or ``name@version``."""
        return self.name if self.version is None else f"{self.name}@{self.version}"

    def allows(self, data: dict[str, Any]) -> bool:
        """Whether a rule (its parsed ``rule.yaml``) applies to this dialect/version.

        :param data: the rule's mapping.
        """
        if self.name not in (data.get("dialects") or []):
            return False
        if self.version is None:
            return True
        key = _version_key(self.name, self.version)
        since = (data.get("since") or {}).get(self.name)
        until = (data.get("until") or {}).get(self.name)
        if since is not None and key < _version_key(self.name, str(since)):
            return False
        return not (until is not None and key >= _version_key(self.name, str(until)))


def _version_key(name: str, version: str) -> tuple[int, ...]:
    if name == "matlab":
        match = _MATLAB_VERSION.match(version)
        if match is None:
            raise DialectError(f"bad matlab version {version!r}; expected e.g. R2023a")
        return (int(match.group(1)), 0 if match.group(2) == "a" else 1)
    if not _OCTAVE_VERSION.match(version):
        raise DialectError(f"bad octave version {version!r}; expected e.g. 8.4")
    return tuple(int(part) for part in version.split("."))


def parse_dialect(spec: str | None) -> Dialect:
    """Parse a ``--dialect`` value such as ``matlab@R2023a``.

    :param spec: the value, or ``None`` for the default (``octave``, any version).
    :raises DialectError: for an unknown dialect or a malformed version.
    """
    if not spec:
        return Dialect()
    name, _, version = spec.partition("@")
    name = name.strip().lower()
    if name not in DIALECTS:
        raise DialectError(f"unknown dialect {name!r}; expected one of {', '.join(DIALECTS)}")
    if version:
        _version_key(name, version)
        return Dialect(name=name, version=version)
    return Dialect(name=name)
