import numpy as np
import pandas as pd
import datetime as dt

input_file = "C:/Users/Administrator/Desktop/BaData/BA_USDNOK_15min.txt"

df = pd.read_csv(input_file, index_col=0, parse_dates=[[0, 1]], delimiter=',')

ret = df['Close'].pct_change()

mon = ret[ret.index.weekday == 1]
mon_grouped = mon.groupby(by = [mon.index.map(lambda x: x.hour),
                                mon.index.map(lambda x: x.minute)])
mon_grouped.plot()
