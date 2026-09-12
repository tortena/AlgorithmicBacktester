import yfinance as yf
import pandas as pd


def get_pairs_data(ticker1, ticker2, start_date, end_date):
    """Fetch historical data from Yahoo Finance."""
    data = yf.download([ticker1, ticker2], start=start_date, end=end_date, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"].copy()
    else:
        closes = data[["Close"]].copy()
        closes.columns = [ticker1]

    closes = closes.dropna()
    closes.columns = [ticker1, ticker2]
    return closes

def split_is_oos(df, split_date):
    """
    Split a DataFrame indexed by date into in-sample and out-of-sample halves.

    Returns
    -------
    (df_is, df_oos)
    """
    df_is = df.loc[df.index < split_date]
    df_oos = df.loc[df.index >= split_date]

    return (df_is, df_oos)