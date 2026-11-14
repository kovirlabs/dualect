# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Executing both sides of a translation pair and collecting their variables.

A script's *outputs* are the variables it leaves in its workspace. Both runners
return them as a ``{name: value}`` mapping so :mod:`dualect.compare` can match
them by name.

The ``.m`` side runs in GNU Octave through oct2py. Starting Octave takes about a
second, so :class:`MRunner` keeps one session and clears it between scripts.
"""

import contextlib
import io
import os
import shutil
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from dualect.errors import ExecutionError, RuntimeMissingError

#: Environment variable oct2py reads to find the interpreter; we honour it too.
EXECUTABLE_ENV = "OCTAVE_EXECUTABLE"

#: Names never treated as outputs: Octave's implicit result variable.
_M_IMPLICIT = frozenset({"ans"})


@dataclass(frozen=True)
class CallableValue:
    """Stand-in for a function handle or lambda.

    Behaviour of a callable cannot be compared by value; case authors compare
    the results of calling it instead. The marker lets both sides agree that a
    name holds *a* callable without pretending the bodies were checked.
    """

    description: str


def m_runtime_available() -> bool:
    """Return whether an Octave executable can be found for running ``.m`` code."""
    configured = os.environ.get(EXECUTABLE_ENV)
    if configured:
        return Path(configured).is_file() or shutil.which(configured) is not None
    return shutil.which("octave") is not None or shutil.which("octave-cli") is not None


def _quote_m_string(text: str) -> str:
    """Quote text as a single-quoted .m string literal (quotes are doubled)."""
    return "'" + text.replace("'", "''") + "'"


class MRunner:
    """Runs ``.m`` scripts in one reusable Octave session.

    Use as a context manager, or call :meth:`close` when done::

        with MRunner() as runner:
            outputs = runner.run(path=Path("case.m"))
    """

    def __init__(self) -> None:
        """Create the runner; Octave itself starts lazily on the first run."""
        self._session: Any = None

    def _ensure_session(self) -> Any:
        """Start Octave on first use, with a clear error when it is missing."""
        if self._session is None:
            if not m_runtime_available():
                raise RuntimeMissingError(
                    "GNU Octave was not found on PATH (or via OCTAVE_EXECUTABLE); "
                    "install it to run .m code (macOS: brew install octave; "
                    "Ubuntu: apt install octave)"
                )
            from oct2py import Oct2Py  # deferred: importing is cheap, starting is not

            self._session = Oct2Py()
        return self._session

    def run(self, *, path: Path) -> dict[str, Any]:
        """Execute a ``.m`` script in a cleared workspace and return its variables.

        The script runs with ``source`` from its own folder, so it may define
        functions before use (with a leading ``1;``) and read neighbouring files.

        :param path: the script.
        :returns: variable name -> value. Logical arrays come back as NumPy
            ``bool`` arrays and function handles as :class:`CallableValue`.
        :raises ExecutionError: if Octave reports an error.
        """
        from oct2py import Oct2PyError

        session = self._ensure_session()
        path = path.resolve()
        printed: list[str] = []
        try:
            # Change directory first: the previous script's folder may have been
            # a temporary one that is gone now, and Octave refuses to run
            # anything from a working directory that no longer exists.
            session.eval(f"cd({_quote_m_string(str(path.parent))});", stream_handler=printed.append)
            # A fresh workspace per script: leftovers from an earlier script
            # (variables or command-line functions) would otherwise leak in.
            session.eval("clear all; close all; more off;", stream_handler=printed.append)
            session.eval(f"source({_quote_m_string(str(path))});", stream_handler=printed.append)
            return self._collect(session=session)
        except Oct2PyError as exc:
            raise ExecutionError(side="m", path=path, message=str(exc)) from exc

    @staticmethod
    def _collect(*, session: Any) -> dict[str, Any]:
        """Pull every workspace variable, using its Octave class to decode it."""
        names = session.eval("who();", nout=1)
        if isinstance(names, str):
            names = [names]
        outputs: dict[str, Any] = {}
        for name in (str(n) for n in np.asarray(names, dtype=object).ravel()):
            if name in _M_IMPLICIT:
                continue
            m_class = session.eval(f"class({name});", nout=1)
            if m_class == "function_handle":
                outputs[name] = CallableValue(description=str(session.pull(name)))
                continue
            value = session.pull(name)
            # oct2py transfers logicals as uint8; restore the type so a
            # logical-vs-double mix-up between the sides is detectable.
            if m_class == "logical":
                value = np.asarray(value, dtype=bool)
            outputs[name] = value
        return outputs

    def close(self) -> None:
        """Stop the Octave session, if one was started."""
        if self._session is not None:
            self._session.exit()
            self._session = None

    def __enter__(self) -> Self:
        """Support ``with MRunner() as runner:``."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Stop Octave when the ``with`` block ends."""
        self.close()


def run_py_script(*, path: Path) -> dict[str, Any]:
    """Execute a Python script in a fresh namespace and return its variables.

    The script runs in this process (it is code the user asked to verify).
    Standard output is captured and discarded so verification output stays clean.

    :param path: the script.
    :returns: variable name -> value. Names starting with ``_``, modules,
        classes and ``def`` functions are dropped (they are scaffolding, not
        outputs); lambdas become :class:`CallableValue`.
    :raises ExecutionError: if the script raises.
    """
    path = path.resolve()
    source = path.read_text(encoding="utf-8")
    namespace: dict[str, Any] = {"__name__": "__main__", "__file__": str(path)}
    try:
        code = compile(source, str(path), "exec")
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, namespace)
    except Exception as exc:  # any failure of user code is reported, not raised raw
        last = traceback.format_exception_only(exc)[-1].strip()
        raise ExecutionError(side="py", path=path, message=last) from exc

    outputs: dict[str, Any] = {}
    for name, value in namespace.items():
        if name.startswith("_") or isinstance(value, types.ModuleType | type):
            continue
        # Any other callable (def functions, imported functions, ufuncs) is a
        # tool the script used, not a result it produced.
        if callable(value):
            if isinstance(value, types.FunctionType) and value.__name__ == "<lambda>":
                outputs[name] = CallableValue(description="lambda")
            continue
        outputs[name] = value
    return outputs
