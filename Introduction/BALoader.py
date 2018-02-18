import pandas as pd
from datetime import datetime

# ---------------------------
#  Configs - User adjust
#
input_file = "C:/Users/David Jaggi/Google Drive/Eiffeltower Capital/Data/AUDCAD.FXCM.asc"
ba_ready_file = "C:/Users/David Jaggi/Google Drive/Eiffeltower Capital/BaData/BA_AUDCAD.txt"

# ----------------------
#  Do NOT Change
#
df = pd.read_csv(input_file, delimiter=',', index_col= 0, parse_dates=[[0,1]])

fh = open(ba_ready_file, 'w')
fh.write("Date,Time,Open,High,Low,Close,Vol,OI\n")
for d,o,h,l,c,v in zip(df.index,df['Open'],df['High'],df['Low'],df['Close'],df['TotalVolume']):
 fh.write("%s,%s,%.5f,%.5f,%.5f,%.5f,%d,%d\n" % (d.strftime('%m/%d/%Y'),d.strftime('%H:%M'),o,h,l,c,v,0))
fh.close()