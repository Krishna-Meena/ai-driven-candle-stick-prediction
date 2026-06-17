from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class BacktestTrade:
    entry_date: datetime
    exit_date: datetime
    side: str
    entry_price: float
    exit_price: float
    return_pct: float
    won: bool


@dataclass(frozen=True)
class BacktestResult:
    symbol: str
    model_label: str
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return_pct: float
    strategy_return_pct: float
    buy_hold_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    equity_dates: list[datetime] = field(default_factory=list)
    equity_values: list[float] = field(default_factory=list)
    monthly_returns: dict[str, float] = field(default_factory=dict)
    trades: list[BacktestTrade] = field(default_factory=list)
    initial_capital: float = 10000.0
    final_equity: float = 10000.0
