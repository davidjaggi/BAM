# https://docs.mongodb.com/tutorials/install-mongodb-on-windows/
# https://github.com/manahl/arctic/blob/master/howtos/201507_demo_pydata.py
from arctic import Arctic
from arctic import TICK_STORE
import pandas as pd
import pytz as tz
import datetime as dt

# Connect to local mongodb
arctic = Arctic('localhost')

# Import file from csv
file = "C:/Users/Administrator/Google Drive/Eiffeltower Capital/BaData/BA_EURUSD_15min.txt"
df = pd.read_csv(file, delimiter=',', index_col = 0, parse_dates=[[0,1]])
df = df.tz_localize('UTC')
# Show head of file
df.head()
df.index
# Initialize the library
arctic.delete_library('FX')
arctic.initialize_library('FX')
arctic.list_libraries()

fx = arctic['FX']

# Write as EURUSD
fx.write('EURUSD', df, metadata={'source':'BuildAlpha', 'timezone':'UTC'})

# Read library EURUSD
item = fx.read('EURUSD')
fx.read('EURUSD').metadata
fx.read('EURUSD').version

# Extract data
eurusd = item.data

# Extract metadata
metadata = item.metadata

# list symbols in library
library.list_symbols()

# get number of datapoints in storage
store.get_quota('FX')

# delete
library.delete('EURUSD')