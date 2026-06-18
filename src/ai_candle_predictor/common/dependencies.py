from __future__ import annotations

from typing import NamedTuple


class OptionalDependency(NamedTuple):
    name: str
    import_name: str
    extra: str


_OPTIONAL: list[OptionalDependency] = [
    OptionalDependency("optuna", "optuna", "ml"),
    OptionalDependency("lightgbm", "lightgbm", "ml"),
    OptionalDependency("mlflow", "mlflow", "ml"),
]


def check_optional(extra: str = "") -> list[dict[str, object]]:
    """Check which optional dependencies are available.

    Returns a list of dicts with keys: ``name``, ``available``, ``install_hint``.
    If *extra* is provided, only deps from that extra group are checked.
    """
    results: list[dict[str, object]] = []
    for dep in _OPTIONAL:
        if extra and extra not in dep.extra:
            continue
        try:
            __import__(dep.import_name)
            available = True
        except ModuleNotFoundError:
            available = False
        results.append(
            {
                "name": dep.name,
                "available": available,
                "install_hint": f"uv add {dep.name}",
            }
        )
    return results
