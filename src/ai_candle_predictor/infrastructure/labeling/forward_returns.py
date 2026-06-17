from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ai_candle_predictor.common.logging import get_logger
from ai_candle_predictor.domain.entities.candle import CandleStick
from ai_candle_predictor.domain.entities.label import Label, LabeledSample

log = get_logger(__name__)


def generate_labels(
    candles: Sequence[CandleStick],
    horizon: int = 5,
    threshold: float = 0.005,
) -> list[LabeledSample]:
    if len(candles) < horizon + 1:
        log.warning(
            "not enough candles for label generation",
            required=horizon + 1,
            actual=len(candles),
        )
        return []

    closes = np.array([c.close for c in candles])
    timestamps = [c.timestamp for c in candles]
    symbol = candles[0].symbol

    forward_returns = _compute_forward_returns(closes, horizon)

    samples: list[LabeledSample] = []
    for i, fr in enumerate(forward_returns):
        if np.isnan(fr):
            continue

        label = _classify(fr, threshold)
        if label == Label.EXCLUDED:
            continue

        samples.append(
            LabeledSample(
                symbol=symbol,
                timestamp=timestamps[i],
                label=label,
                forward_return=float(fr),
                horizon=horizon,
                close=float(closes[i]),
            )
        )

    log.info(
        "labels generated",
        symbol=symbol,
        horizon=horizon,
        threshold=threshold,
        total=len(samples),
    )
    return samples


def generate_labels_raw(
    candles: Sequence[CandleStick],
    horizon: int = 5,
    threshold: float = 0.005,
) -> list[LabeledSample]:
    if len(candles) < horizon + 1:
        return []

    closes = np.array([c.close for c in candles])
    timestamps = [c.timestamp for c in candles]
    symbol = candles[0].symbol

    forward_returns = _compute_forward_returns(closes, horizon)

    samples: list[LabeledSample] = []
    for i, fr in enumerate(forward_returns):
        if np.isnan(fr):
            continue
        label = _classify(fr, threshold)
        samples.append(
            LabeledSample(
                symbol=symbol,
                timestamp=timestamps[i],
                label=label,
                forward_return=float(fr),
                horizon=horizon,
                close=float(closes[i]),
            )
        )

    return samples


def _compute_forward_returns(closes: NDArray[np.float64], horizon: int) -> NDArray[np.float64]:
    shifted = np.roll(closes, -horizon)
    shifted[-horizon:] = np.nan
    return (shifted - closes) / closes


def _classify(forward_return: float, threshold: float) -> Label:
    if forward_return > threshold:
        return Label.UP
    if forward_return < -threshold:
        return Label.DOWN
    return Label.EXCLUDED
