import math
import numpy as np

def performance_summary(daily_pnl, positions, periods_per_year=252):
    """
    Returns
    -------
    dict with keys: 'sharpe', 'max_drawdown', 'num_trades', 'win_rate'
    """

    sharpe = daily_pnl.mean() / daily_pnl.std() * math.sqrt(periods_per_year)
    cumulative = daily_pnl.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative  - running_max
    max_drawdown = abs(drawdown.min())

    # Only approximate: excludes 1 -> -1.
    num_trades = ((positions==0).shift(1) & positions!=0).sum()
    win_rate = (0<daily_pnl).sum()/(daily_pnl[~np.isnan(daily_pnl)]!=0).sum()

    return {
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "num_trades": num_trades,
        "win_rate": win_rate
    }