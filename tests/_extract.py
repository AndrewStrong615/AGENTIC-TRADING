"""Test-only helper for pulling individual pure functions out of
``backtest.py`` without importing that module.

``backtest.py`` imports ``Tradingcore.py`` (and, transitively, heavy/optional
dependencies such as scikit-learn, hmmlearn, vaderSentiment and
alpaca_trade_api) purely to build the live trading engine. A plain
``import backtest`` therefore pulls in a large, sometimes-unavailable
dependency chain just to reach a couple of pure math functions, and in some
Python versions it can even fail to import at all for reasons that have
nothing to do with the functions under test (e.g. syntax that requires a
newer interpreter).

To keep the unit tests hermetic, fast, and independent of that import chain,
we parse the target source file with ``ast`` and re-execute *only* the named
top-level function definitions (plus whatever extra source/globals the
caller supplies) in an isolated namespace. This never imports backtest.py,
Tradingcore.py, bot.py, or credentials_store.py, and never triggers any of
their module-level side effects (network calls, credential loading, etc).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable


def extract_functions(
    file_path: str | Path,
    function_names: Iterable[str],
    extra_globals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a namespace dict containing only the requested top-level
    functions from ``file_path``, executed with ``extra_globals`` (plus
    builtins) as their globals.

    Raises AssertionError if any requested function name is not found as a
    top-level ``def`` in the file, so a rename in the source file fails the
    test suite loudly instead of silently testing nothing.
    """
    path = Path(file_path)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    wanted = set(function_names)
    found_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    found_names = {node.name for node in found_nodes}
    missing = wanted - found_names
    assert not missing, (
        f"function(s) {sorted(missing)} not found as top-level defs in "
        f"{path} — the source may have been renamed/refactored; update the "
        f"test accordingly"
    )

    namespace: dict[str, Any] = dict(extra_globals or {})
    namespace.setdefault("__name__", f"_extracted_from_{path.stem}")

    module = ast.Module(body=found_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    code = compile(module, filename=str(path), mode="exec")
    exec(code, namespace)  # noqa: S102 - controlled, test-only source subset

    return namespace
