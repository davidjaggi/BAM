import pandas as pd
import numpy as np


input_file = "C:/Users/Administrator/Desktop/Data Quote Manager/AUDCAD.FXCM-Minute-Trade.asc"
ba_ready_file = "C:/Users/Administrator/Desktop/BaData/BA_AUDCAD_15min.txt"

df = pd.read_csv(input_file,index_col=0, parse_dates=[[0,1]])


df_resample = df.resample('15Min').agg({'Open': 'first',
                                 'High': 'max',
                                 'Low': 'min',
                                 'Close': 'last',
                                        'TotalVolume': 'sum'})

fh = open(ba_ready_file, 'w')
fh.write("Date,Time,Open,High,Low,Close,Vol,OI\n")
for d,o,h,l,c,v in zip(df_resample.index,df_resample['Open'],df_resample['High'],df_resample['Low'],df_resample['Close'],df_resample['TotalVolume']):
 fh.write("%s,%s,%.5f,%.5f,%.5f,%.5f,%.2f,%d\n" % (d.strftime('%m/%d/%Y'),d.strftime('%H:%M'),o,h,l,c,v,0))
fh.close()


df.head()
df_open =  df['Open'].resample('15Min').ohlc()
df_high =  df['High'].resample('15Min').ohlc()
df_low =  df['Low'].resample('15Min').ohlc()
df_close =  df['Close'].resample('15Min').ohlc()
df_vol =  df['TotalVolume'].resample('15Min').ohlc()

df_resample = pd.concat([df_open,df_high,df_low,df_close])
df_resample = df_resample.dropna()



# ----------------------
#  Do NOT Change
#
fh = open(ba_ready_file, 'w')
fh.write("Date,Time,Open,High,Low,Close,Vol,OI\n")
for d,o,h,l,c in zip(df_resample.index,df_resample['open'],df_resample['high'],df_resample['low'],df_resample['close']):
 fh.write("%s,%s,%.5f,%.5f,%.5f,%.5f,%d,%d\n" % (d.strftime('%m/%d/%Y'),d.strftime('%H:%M'),o,h,l,c,0,0))
fh.close()
