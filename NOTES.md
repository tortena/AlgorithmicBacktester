RUN PROJECT WITH run_backtest.py

### Extensions
Vectorising calculations - speed up
Use OU process to justify lookback window
Take as an input many stocks, and adf_test combinations


### Preventing lookback bias
Remember that a day's stock value is found at the end of the day. If we use yesterday's stock value to determine the hedge ratio from yesterday, then our position that uses that hedge ratio is only applicable today

### Preventing data snooping
Chose split date at the beginning of the project
