from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    support: int

    def __repr__(self) -> str:
        return (
            f"ClassificationMetrics(\n"
            f"  accuracy ={self.accuracy:.4f}\n"
            f"  precision={self.precision:.4f}\n"
            f"  recall   ={self.recall:.4f}\n"
            f"  f1       ={self.f1:.4f}\n"
            f"  roc_auc  ={self.roc_auc:.4f}\n"
            f"  support  ={self.support}\n"
            f")"
        )
