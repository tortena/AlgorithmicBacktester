import numpy as np
import pandas as pd


def generate_signals(z_scores, entry_threshold=2.0, exit_threshold=0.5):
    """
    Convert a z-score series into a position series.

    Position convention:
      +1  = long the spread   (long asset1, short hedge_ratio * asset2)
      -1  = short the spread  (short asset1, long hedge_ratio * asset2)
       0  = flat

    Logic:
      - If flat and z < -entry_threshold  -> go long
      - If flat and z >  entry_threshold  -> go short
      - If in a position and |z| < exit_threshold -> go flat
      - Otherwise, hold the current position

    Returns
    -------
    pd.Series of positions {-1, 0, 1}, same index as z_scores.
    """

    FLAT=0
    LONG_SPREAD = 1
    SHORT_SPREAD = -1

    
    positions = pd.Series(0, index=z_scores.index, dtype=int)

    curr_pos = FLAT

    for i in range(positions.shape[0]):
        if np.isnan(z_scores.iloc[i]):
            positions.iloc[i] = FLAT
            curr_pos = FLAT

            
        z = z_scores.iloc[i]
        if curr_pos == FLAT:           
            if z > entry_threshold: 
                curr_pos = SHORT_SPREAD
            elif z < -entry_threshold:
                curr_pos = LONG_SPREAD

        elif curr_pos == SHORT_SPREAD and z < exit_threshold:
            curr_pos = FLAT
        elif curr_pos == LONG_SPREAD and z > -exit_threshold:
            curr_pos = FLAT

        positions.iloc[i] = curr_pos

    return positions


