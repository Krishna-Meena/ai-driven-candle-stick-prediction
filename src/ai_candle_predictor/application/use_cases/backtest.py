from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np

from ai_candle_predictor.application.dto.backtest import BacktestResult, BacktestTrade
from ai_candle_predictor.application.dto.prediction import CandlePrediction
from ai_candle_predictor.common.logging import get_logger

log = get_logger(__name__)


def run_backtest(
    predictions: Sequence[CandlePrediction],
    initial_capital: float = 10000.0,
    _horizon: int = 5,
) -> BacktestResult:
    if not predictions:
        return BacktestResult(
            symbol="",
            model_label="",
            start_date=datetime.min,
            end_date=datetime.min,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_return_pct=0.0,
            strategy_return_pct=0.0,
            buy_hold_return_pct=0.0,
            sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
        )

    sorted_preds = sorted(predictions, key=lambda p: p.timestamp)

    equity = float(initial_capital)
    peak = equity
    equity_curve_dates: list[datetime] = []
    equity_curve_values: list[float] = []
    trades: list[BacktestTrade] = []
    daily_returns: list[float] = []
    in_position = False
    position_side = ""
    entry_price = 0.0
    entry_date: datetime | None = None

    for p in sorted_preds:
        pred_up = p.predicted_direction == 1
        close_price = p.close
        ret = p.actual_return if p.actual_return is not None else 0.0
        sign = 1.0 if pred_up else -1.0
        strategy_ret = sign * ret
        daily_returns.append(strategy_ret)

        equity *= 1.0 + strategy_ret
        peak = max(peak, equity)
        equity_curve_dates.append(p.timestamp)
        equity_curve_values.append(equity)

        if pred_up and not in_position:
            in_position = True
            position_side = "LONG"
            entry_price = close_price
            entry_date = p.timestamp
        elif not pred_up and in_position:
            in_position = False
            if position_side == "LONG":
                trade_ret = (close_price - entry_price) / entry_price
            else:
                trade_ret = (entry_price - close_price) / entry_price
            trades.append(
                BacktestTrade(
                    entry_date=entry_date or p.timestamp,
                    exit_date=p.timestamp,
                    side=position_side,
                    entry_price=entry_price,
                    exit_price=close_price,
                    return_pct=trade_ret * 100,
                    won=trade_ret > 0,
                )
            )
            position_side = ""
            entry_price = 0.0
            entry_date = None

    if in_position:
        last = sorted_preds[-1]
        if position_side == "LONG":
            trade_ret = (last.close - entry_price) / entry_price
        else:
            trade_ret = (entry_price - last.close) / entry_price
        trades.append(
            BacktestTrade(
                entry_date=entry_date or last.timestamp,
                exit_date=last.timestamp,
                side=position_side,
                entry_price=entry_price,
                exit_price=last.close,
                return_pct=trade_ret * 100,
                won=trade_ret > 0,
            )
        )

    total_return_pct = ((equity - initial_capital) / initial_capital) * 100
    buy_hold_rets = [p.actual_return if p.actual_return is not None else 0.0 for p in sorted_preds]
    buy_hold_equity = float(initial_capital)
    for r in buy_hold_rets:
        buy_hold_equity *= 1.0 + r
    buy_hold_return_pct = ((buy_hold_equity - initial_capital) / initial_capital) * 100

    winning = sum(1 for t in trades if t.won)
    losing = sum(1 for t in trades if not t.won)
    total_trades = len(trades)
    win_rate = winning / total_trades if total_trades > 0 else 0.0

    max_dd = _compute_max_drawdown(equity_curve_values)

    sharpe = _compute_sharpe(daily_returns)

    monthly = _compute_monthly_returns(sorted_preds)

    return BacktestResult(
        symbol=sorted_preds[0].timestamp.strftime("%Y-%m-%d"),
        model_label="",
        start_date=sorted_preds[0].timestamp,
        end_date=sorted_preds[-1].timestamp,
        total_trades=total_trades,
        winning_trades=winning,
        losing_trades=losing,
        win_rate=win_rate,
        total_return_pct=total_return_pct,
        strategy_return_pct=total_return_pct,
        buy_hold_return_pct=buy_hold_return_pct,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        equity_dates=equity_curve_dates,
        equity_values=equity_curve_values,
        monthly_returns=monthly,
        trades=trades,
        initial_capital=initial_capital,
        final_equity=equity,
    )


def _compute_max_drawdown(equity_values: list[float]) -> float:
    if len(equity_values) < 2:
        return 0.0
    peak = equity_values[0]
    max_dd = 0.0
    for v in equity_values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd * 100


def _compute_sharpe(daily_returns: list[float], risk_free: float = 0.0) -> float:
    if len(daily_returns) < 2:
        return 0.0
    arr = np.array(daily_returns, dtype=np.float64)
    excess = arr - risk_free
    if excess.std(ddof=1) == 0:
        return 0.0
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(252))


def _compute_monthly_returns(predictions: Sequence[CandlePrediction]) -> dict[str, float]:
    monthly: dict[str, list[float]] = {}
    for p in predictions:
        key = p.timestamp.strftime("%Y-%m")
        ret = p.actual_return if p.actual_return is not None else 0.0
        sign = 1.0 if p.predicted_direction == 1 else -1.0
        monthly.setdefault(key, []).append(sign * ret)
    return {k: float(np.mean(v) * 100) for k, v in monthly.items()}
