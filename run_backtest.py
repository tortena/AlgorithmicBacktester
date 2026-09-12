from data import get_pairs_data, split_is_oos
from stats import adf_test, rolling_spread, rolling_hedge_ratio, calculate_zscore
from signals import generate_signals
from engine import run_backtest, grid_search_thresholds
from performance_summary import performance_summary

import pandas as pd
import numpy as np

TRANSACTION_COST = 0.0
ADF_SIGNIFICANCE = 0.15 # Default should be 0.05
LOOKBACK = 60 #Can be replaced for an analytical selection with an OU process
# periods per year = 252 by default
SPLIT_DATE="2025-01-01"


df = get_pairs_data("KO", "PEP", "2024-01-01", "2026-01-01")
df_is, df_oos = split_is_oos(df, SPLIT_DATE) # SPLIT_DATE must be in oos, excluded from is

hedge_ratios_is = rolling_hedge_ratio(df_is["KO"], df_is["PEP"])
spread_is = rolling_spread(df_is["KO"], df_is["PEP"], hedge_ratios_is)

adf_res_dict = adf_test(spread_is, significance=ADF_SIGNIFICANCE) # Adjust significance

if adf_res_dict["is_stationary"] == False:
    print("Spread is not stationary for in_sample")
    print(adf_res_dict)
else:
    z_scores_is = calculate_zscore(spread_is, lookback=LOOKBACK)

    # Adjust threshold grid + transaction cost
    threshold_scores = grid_search_thresholds(df_is["KO"], df_is["PEP"], hedge_ratios_is, z_scores_is, transaction_cost=TRANSACTION_COST)

    # Thresholds which have only a few trades can have high Sharpe by luck
    high_trade_thresholds = threshold_scores[threshold_scores["num_trades"] >= 10]

    if len(high_trade_thresholds)==0:
        # Have no choice to use low-trade strategies
        print("WARNING: Low trade_num thresholds chosen")
        high_trade_thresholds = threshold_scores

    highest_sharpe_row = high_trade_thresholds.loc[high_trade_thresholds["sharpe"].idxmax()]

    chosen_entry_threshold = highest_sharpe_row["entry"]
    chosen_exit_threshold = highest_sharpe_row["exit"]

    # Test on out-of-sample

    # Concatenation prevents initial oos values being filled with unnecessary np.nan values
    hedge_ratios_full = rolling_hedge_ratio(df["KO"], df["PEP"])
    spread_full = rolling_spread(df["KO"], df["PEP"], hedge_ratios_full)
    z_scores_full = calculate_zscore(spread_full, lookback=LOOKBACK)

    hedge_ratios_oos = split_is_oos(hedge_ratios_full,SPLIT_DATE)[1]
    spread_oos = split_is_oos(spread_full,SPLIT_DATE)[1]
    z_scores_oos = split_is_oos(z_scores_full,SPLIT_DATE)[1]

    positions_oos = generate_signals(z_scores_oos, 
                                     entry_threshold=chosen_entry_threshold,
                                     exit_threshold=chosen_exit_threshold)

    backtest_res = run_backtest(df_oos["KO"], df_oos["PEP"], hedge_ratios_oos, positions_oos, transaction_cost=TRANSACTION_COST)
    performance_res = performance_summary(backtest_res["daily_pnl"], positions_oos)

    print()
    print(f"Entry_threshold = {chosen_entry_threshold}")
    print(f"Exit_threshold = {chosen_exit_threshold}")
    print()
    print(performance_res)
    print()
    print(f"Total cumulative pnl: {backtest_res["cumulative_pnl"].iloc[-1]} trade units")

