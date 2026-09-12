import numpy as np
import pandas as pd
from signals import generate_signals
from performance_summary import performance_summary

def run_backtest(asset1, asset2, hedge_ratios, positions, transaction_cost=0.0):
    """
    Simulate a dollar-neutral pairs trade.

    For each day, if position == +1: you are long 1 unit of asset1 and
    short hedge_ratio units of asset2 (and the mirror image for -1).

    Parameters
    ----------
    asset1, asset2 : pd.Series of prices
    hedge_ratios : pd.Series (from rolling_hedge_ratio)
    positions : pd.Series of {-1, 0, 1} (from generate_signals)
    transaction_cost : float
        Cost per unit traded, applied when the position CHANGES.

    Returns
    -------
    pd.DataFrame with columns: ['daily_pnl', 'cumulative_pnl']
    """

    daily_pnl = pd.Series(0, index=asset1.index, dtype=np.float64)
    prev_pos = positions.iloc[0]
    for i in range(1, len(positions)):
        
        curr_pos = positions.iloc[i]
        
        curr_date = positions.index[i]
        prev_date = positions.index[i-1]

        hr = hedge_ratios.loc[prev_date]

        asset1_price_change = asset1.loc[curr_date] - asset1.loc[prev_date]
        asset2_price_change = asset2.loc[curr_date] - asset2.loc[prev_date]

        if np.isnan(prev_pos):
            daily_pnl.iloc[i] = 0
        else:
            daily_pnl.iloc[i] = prev_pos*(asset1_price_change - hr*asset2_price_change)            
            daily_pnl.iloc[i] -= (1+hr)*transaction_cost*abs(curr_pos-prev_pos)
       

        prev_pos = curr_pos
            

    cumulative_pnl = daily_pnl.cumsum()


    return pd.DataFrame({'daily_pnl': daily_pnl, 'cumulative_pnl': cumulative_pnl})


def grid_search_thresholds(asset1_is, asset2_is, hedge_ratios_is, z_scores_is, transaction_cost=0.0,
                           entry_grid=(1.5,2.0,2.5), exit_grid=(0.25, 0.5, 0.75)):

    """
    Try each (entry, exit) combination on IN-SAMPLE data only.

    Small grids prevent overtraining

    Returns
    -------
    pd.DataFrame with columns ['entry', 'exit', 'sharpe', 'num_trades'],
    one row per combination tried.
    """

    performance_results = []

    for entry in entry_grid:
        for exit in exit_grid:
            positions = generate_signals(z_scores_is, entry_threshold=entry, exit_threshold=exit)

            backtest_results = run_backtest(asset1_is, asset2_is, hedge_ratios_is, positions, transaction_cost=transaction_cost)

            performance_res = performance_summary(backtest_results["daily_pnl"], positions)

            performance_results.append([entry, exit, performance_res["sharpe"], performance_res["num_trades"]])

    return  pd.DataFrame(performance_results, columns=["entry", "exit", "sharpe", "num_trades"])

