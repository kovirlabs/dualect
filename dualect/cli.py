# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""The ``dualect`` command (docs/design.md section 8).

::

    dualect scan script.m [--json]         # which rules does this file need?
    dualect context script.m               # LLM-ready context for those rules
    dualect translate script.m -o out.py [--strict] [--dialect matlab@R2023a]
    dualect verify script.m script.py      # run both, compare
    dualect rules list [--status guided]
    dualect rules new indexing/foo

Exit codes: 0 success; 1 verification or validation failed, ``--strict``
stopped, or an error was reported; 2 bad command-line usage (argparse);
3 command not implemented yet; 4 ``translate`` wrote an incomplete or
unverified translation (its report and first line say which).
"""

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from pathlib import Path

from dualect import __version__
from dualect.compare import DEFAULT_ATOL, DEFAULT_RTOL, Tolerance
from dualect.context import build_context
from dualect.dialect import parse_dialect
from dualect.errors import DualectError
from dualect.paths import default_rules_dir
from dualect.pipeline import StrictModeError, translate_file
from dualect.rulebook import Status, load_rules
from dualect.scaffold import new_rule
from dualect.scan import ScanResult, scan_file
from dualect.verify import options_for_pair, verify_pair

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_NOT_IMPLEMENTED = 3
EXIT_INCOMPLETE = 4


def _scan_source(args: argparse.Namespace) -> ScanResult:
    """Scan the file named on the command line with the chosen rules and dialect."""
    rules = load_rules(rules_dir=args.rules_dir)
    return scan_file(path=args.source, rules=rules, dialect=parse_dialect(args.dialect))


def _scan(args: argparse.Namespace) -> int:
    """Handle ``dualect scan``."""
    result = _scan_source(args)
    print(result.to_json() if args.json else result.report())
    return EXIT_OK


def _context(args: argparse.Namespace) -> int:
    """Handle ``dualect context``."""
    print(build_context(result=_scan_source(args)), end="")
    return EXIT_OK


def _translate(args: argparse.Namespace) -> int:
    """Handle ``dualect translate``: translate, write, verify, report."""
    rules = load_rules(rules_dir=args.rules_dir)
    try:
        result = translate_file(
            source=args.source,
            output=args.output,
            rules=rules,
            dialect=parse_dialect(args.dialect),
            strict=args.strict,
            verify=not args.no_verify,
        )
    except StrictModeError as exc:
        print(f"dualect: {exc}", file=sys.stderr)
        print("dualect: nothing was written", file=sys.stderr)
        return EXIT_FAILED
    print(result.report())
    print(f"wrote {args.output}")
    if result.verified:
        return EXIT_OK
    if result.verify is not None:
        return EXIT_FAILED
    return EXIT_INCOMPLETE


def _verify(args: argparse.Namespace) -> int:
    """Handle ``dualect verify``."""
    options = options_for_pair(
        m_path=args.m_file,
        py_path=args.py_file,
        tolerance=Tolerance(rtol=args.rtol, atol=args.atol),
        extra_index=args.index,
    )
    if args.ignore:
        options = dataclasses.replace(options, ignore=options.ignore | frozenset(args.ignore))
    result = verify_pair(m_path=args.m_file, py_path=args.py_file, options=options)
    print(result.report())
    return EXIT_OK if result.ok else EXIT_FAILED


def _rules_list(args: argparse.Namespace) -> int:
    """Handle ``dualect rules list``."""
    status = Status(args.status) if args.status else None
    rules = load_rules(rules_dir=args.rules_dir, status=status)
    if not rules:
        print("no rules found", file=sys.stderr)
        return EXIT_OK
    width = max(len(rule.id) for rule in rules)
    for rule in rules:
        print(
            f"{rule.id:<{width}}  {rule.status.value:<11}  {len(rule.cases):>2} cases  {rule.title}"
        )
    return EXIT_OK


def _rules_new(args: argparse.Namespace) -> int:
    """Handle ``dualect rules new``."""
    rule_dir = new_rule(rules_dir=args.rules_dir, rule_id=args.rule_id)
    print(f"created {rule_dir}")
    print("next: fill in rule.yaml and cases/basic.m + basic.py, remove every TODO")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (exposed for tests and documentation)."""
    parser = argparse.ArgumentParser(
        prog="dualect",
        description="Translate numeric code between .m files and Python/NumPy, verified.",
    )
    parser.add_argument("--version", action="version", version=f"dualect {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add_source_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("source", type=Path, help="the .m file")
        sub.add_argument(
            "--dialect", help="source dialect and version, e.g. octave or matlab@R2023a"
        )
        sub.add_argument("--rules-dir", type=Path, default=None, help=argparse.SUPPRESS)

    scan = commands.add_parser("scan", help="list the rules a file needs, and what has none")
    add_source_options(scan)
    scan.add_argument("--json", action="store_true", help="machine-readable output")
    scan.set_defaults(handler=_scan)

    context = commands.add_parser("context", help="print LLM-ready context for a file's rules")
    add_source_options(context)
    context.set_defaults(handler=_context)

    translate = commands.add_parser(
        "translate", help="translate a .m file to Python/NumPy, then verify it by execution"
    )
    add_source_options(translate)
    translate.add_argument("-o", "--output", type=Path, required=True, help="the .py to write")
    translate.add_argument(
        "--strict",
        action="store_true",
        help="stop, writing nothing, if any statement needs guided or unsupported translation",
    )
    translate.add_argument(
        "--no-verify", action="store_true", help="skip running both sides (output is UNVERIFIED)"
    )
    translate.set_defaults(handler=_translate)

    verify = commands.add_parser("verify", help="run a .m and a .py script and compare outputs")
    verify.add_argument("m_file", type=Path)
    verify.add_argument("py_file", type=Path)
    verify.add_argument("--rtol", type=float, default=DEFAULT_RTOL, help="relative tolerance")
    verify.add_argument("--atol", type=float, default=DEFAULT_ATOL, help="absolute tolerance")
    verify.add_argument(
        "--index",
        action="append",
        default=[],
        metavar="NAME",
        help="variable holding indices (1-based in .m, 0-based in Python); repeatable",
    )
    verify.add_argument(
        "--ignore", action="append", default=[], metavar="NAME", help="skip a variable; repeatable"
    )
    verify.set_defaults(handler=_verify)

    rules = commands.add_parser("rules", help="inspect or create rules")
    rules.add_argument(
        "--rules-dir",
        type=Path,
        default=None,
        help="rules folder (default: the repository's rules/ or $DUALECT_RULES_DIR)",
    )
    rule_commands = rules.add_subparsers(dest="rules_command", required=True, metavar="ACTION")
    listing = rule_commands.add_parser("list", help="list rules with status and case count")
    listing.add_argument("--status", choices=[status.value for status in Status])
    listing.set_defaults(handler=_rules_list)
    create = rule_commands.add_parser("new", help="scaffold rules/<category>/<rule-id>/")
    create.add_argument("rule_id", metavar="CATEGORY/ID")
    create.set_defaults(handler=_rules_new)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the process exit code.

    :param argv: arguments without the program name (default: ``sys.argv[1:]``).
    """
    args = build_parser().parse_args(argv)
    if hasattr(args, "rules_dir") and args.rules_dir is None:
        args.rules_dir = default_rules_dir()
    try:
        return int(args.handler(args))
    except DualectError as exc:
        print(f"dualect: {exc}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
