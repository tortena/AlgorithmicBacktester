import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

def adf_test(spread, significance=0.05):
    """
    Run an Augmented Dickey-Fuller test on a spread series.

    Parameters
    ----------
    spread : pd.Series
        The spread (e.g. from calculate_spread), NaNs dropped.
    significance : float
        p-value threshold for rejecting the null (default 0.05).

    Returns
    -------
    dict with keys: 'test_statistic', 'p_value', 'is_stationary' (bool)
    """
    
    adf_res = adfuller(spread.dropna())
    resultDict = {
        "test_statistic": adf_res[0],
        "p_value": adf_res[1],
        "is_stationary": adf_res[1] < significance
    }

    return resultDict



def calculate_zscore(spread, lookback=60):
    """Calculate the rolling Z-score of the spread."""
    rolling_mean = spread.rolling(window=lookback).mean()
    rolling_std = spread.rolling(window=lookback).std()
    z_score = (spread - rolling_mean) / rolling_std
    return z_score



def rolling_hedge_ratio(asset1, asset2, window=60):
    """
    At each time t, estimate the hedge ratio using only the trailing
    `window` observations up to and including t (never data after t).

    Returns
    -------
    pd.Series of hedge ratios, same index as asset1/asset2, with the
    first `window` entries as NaN (not enough history yet).
    """
    
    hedge_ratios = pd.Series(np.nan, index=range(asset1.shape[0]))
    t=window-1
    asset2_w_constant = sm.add_constant(asset2)
    while t < len(asset1):
        model = sm.OLS(asset1[t-window+1:t+1], asset2_w_constant[t-window+1:t+1]).fit()
        hedge_ratios[t] = model.params.iloc[1]
        t+=1

    hedge_ratios.index = asset1.index

    return hedge_ratios


def rolling_spread(asset1, asset2, hedge_ratios):
    """
    Build the spread using a *different* hedge ratio at every point in time,
    rather than one static hedge ratio for the whole series.
    """
    return asset1 - hedge_ratios * asset2
    

    
    