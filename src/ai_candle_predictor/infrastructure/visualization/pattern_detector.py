from __future__ import annotations

from collections.abc import Sequence

from ai_candle_predictor.domain.entities.candle import CandleStick
from ai_candle_predictor.domain.entities.patterns import CandlePattern, PatternMatch

_DOJI_BODY_THRESHOLD = 0.05


def detect_patterns(candles: Sequence[CandleStick]) -> list[PatternMatch]:
    singles = _detect_single_patterns(candles)
    engulfing = _detect_engulfing(candles)
    return [*singles, *engulfing]


def _detect_single_patterns(candles: Sequence[CandleStick]) -> list[PatternMatch]:
    matches: list[PatternMatch] = []

    for c in candles:
        body = c.body_size
        total = c.range
        if total == 0:
            continue

        body_ratio = body / total
        upper = c.upper_wick
        lower = c.lower_wick

        if body_ratio <= _DOJI_BODY_THRESHOLD:
            if lower >= body * 4 and upper < body:
                matches.append(
                    PatternMatch(
                        pattern=CandlePattern.HAMMER,
                        timestamp=c.timestamp,
                        symbol=c.symbol,
                        description=f"Dragonfly Doji at {c.low:.2f}",
                    )
                )
            elif upper >= body * 4 and lower < body:
                matches.append(
                    PatternMatch(
                        pattern=CandlePattern.SHOOTING_STAR,
                        timestamp=c.timestamp,
                        symbol=c.symbol,
                        description=f"Gravestone Doji at {c.high:.2f}",
                    )
                )
            else:
                matches.append(
                    PatternMatch(
                        pattern=CandlePattern.DOJI,
                        timestamp=c.timestamp,
                        symbol=c.symbol,
                        confidence=0.9,
                        description=f"Doji body/range={body_ratio:.3f}",
                    )
                )
            continue

        if c.is_bearish and lower >= body * 2 and upper <= body * 0.3:
            matches.append(
                PatternMatch(
                    pattern=CandlePattern.HAMMER,
                    timestamp=c.timestamp,
                    symbol=c.symbol,
                    confidence=0.85,
                    description=f"Hammer lower_wick={lower:.2f} body={body:.2f}",
                )
            )

        if c.is_bullish and upper >= body * 2 and lower <= body * 0.3:
            matches.append(
                PatternMatch(
                    pattern=CandlePattern.SHOOTING_STAR,
                    timestamp=c.timestamp,
                    symbol=c.symbol,
                    confidence=0.85,
                    description=f"Shooting Star upper_wick={upper:.2f} body={body:.2f}",
                )
            )

    return matches


def _detect_engulfing(candles: Sequence[CandleStick]) -> list[PatternMatch]:
    matches: list[PatternMatch] = []
    for i in range(1, len(candles)):
        prev = candles[i - 1]
        curr = candles[i]

        prev_body = prev.body_size
        curr_body = curr.body_size

        if prev_body == 0 or curr_body == 0:
            continue

        if (
            prev.is_bearish
            and curr.is_bullish
            and curr.close > prev.open
            and curr.open < prev.close
        ):
            matches.append(
                PatternMatch(
                    pattern=CandlePattern.BULLISH_ENGULFING,
                    timestamp=curr.timestamp,
                    symbol=curr.symbol,
                    confidence=0.8,
                    description=f"Bullish Engulfing at {curr.close:.2f}",
                )
            )

        if (
            prev.is_bullish
            and curr.is_bearish
            and curr.open > prev.close
            and curr.close < prev.open
        ):
            matches.append(
                PatternMatch(
                    pattern=CandlePattern.BEARISH_ENGULFING,
                    timestamp=curr.timestamp,
                    symbol=curr.symbol,
                    confidence=0.8,
                    description=f"Bearish Engulfing at {curr.close:.2f}",
                )
            )

    return matches
