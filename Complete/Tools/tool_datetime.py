import pandas as pd

def next_datetimes(date, steps, freq):
    dates = list()
    for i in range(steps):
        date = pd.date_range(date, periods = 2, freq = freq)[1]
        dates.append(date)
    return pd.to_datetime(dates)