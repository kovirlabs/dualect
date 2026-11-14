# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Loading the per-rule deterministic translators (``rules/**/translate.py``).

A rule's ``translate.py`` is a small module next to its ``rule.yaml``. It
defines ``translate(node, ctx)``, which turns the IR nodes of that rule's
constructs into Python (:mod:`ast` expressions, or :class:`dualect.emit.Line`
lists for statements), using the :class:`dualect.emit.Context` it is given.
Optional hooks:

* ``translate_store(stmt, ctx)``: an assignment *to* the construct
  (indexing: ``x(i) = v``);
* ``translate_multi(stmt, ctx)``: a call with several outputs (``[m, i] = max(v)``);
* ``position_outputs(call, nargout)``: which outputs of a built-in call are
  1-based positions (``find``), so the emitter may store them 0-based;
* ``translate_positions(call, ctx)``: the 0-based form of such a call.

Translators live with their rule so that a language change touches one folder.
The engine only runs the translator of a rule whose status is ``mechanical``.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

from dualect.errors import RuleError
from dualect.rulebook import TRANSLATOR_FILE, Rule


class TranslatorSet:
    """Lazily loaded translators, by rule id."""

    def __init__(self, *, rules: list[Rule]) -> None:
        """Remember where each rule's translator lives.

        :param rules: all loaded rules.
        """
        self._rules = {rule.id: rule for rule in rules}
        self._modules: dict[str, ModuleType | None] = {}

    def get(self, rule_id: str) -> ModuleType | None:
        """The translator module of a rule, or ``None`` when it has none."""
        if rule_id not in self._modules:
            rule = self._rules.get(rule_id)
            self._modules[rule_id] = load_translator(rule) if rule is not None else None
        return self._modules[rule_id]


_SHARED: dict[Path, ModuleType | None] = {}


def shared_translator(rule: Rule) -> ModuleType | None:
    """A rule's translator, loaded once per process (for inference hooks).

    Inference runs on every scan; reloading every rule's module each time
    would cost more than the scan itself.
    """
    path = rule.path / TRANSLATOR_FILE
    if path not in _SHARED:
        _SHARED[path] = load_translator(rule)
    return _SHARED[path]


def load_translator(rule: Rule) -> ModuleType | None:
    """Import ``translate.py`` from a rule folder.

    :param rule: the rule.
    :returns: the module, or ``None`` when the folder has no translator.
    :raises RuleError: if the file does not define ``translate``.
    """
    path = rule.path / TRANSLATOR_FILE
    if not path.is_file():
        return None
    name = "dualect_rules." + rule.id.replace("/", ".").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuleError(f"{rule.id}: cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "translate", None)):
        raise RuleError(f"{rule.id}: {TRANSLATOR_FILE} must define translate(node, ctx)")
    return module
