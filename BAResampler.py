import pandas as pd
import numpy as np


input_file = "C:/Users/Administrator/Desktop/Data Quote Manager/GBPUSD.FXCM-Minute-Trade.asc"
ba_ready_file = "C:/Users/Administrator/Desktop/BaData/BA_GBPUSD_15min.txt"

df = pd.read_csv(input_file,index_col=0, parse_dates=[[0,1]])


df_resample = df.resample('15Min', closed='right', label='right').agg({'Open': 'first',
                                                        'High': 'max',
                                                        'Low': 'min',
                                                        'Close': 'last',
                                                        'TotalVolume': 'sum'})

df_resample = df_resample.dropna()

fh = open(ba_ready_file, 'w')
fh.write("Date,Time,Open,High,Low,Close,Vol,OI\n")
for d,o,h,l,c in zip(df_resample.index,df_resample['Open'],df_resample['High'],df_resample['Low'],df_resample['Close'],):
 fh.write("%s,%s,%.5f,%.5f,%.5f,%.5f,%d,%d\n" % (d.strftime('%m/%d/%Y'),d.strftime('%H:%M'),o,h,l,c,0,0))
fh.close()