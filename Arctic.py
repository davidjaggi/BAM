# https://docs.mongodb.com/tutorials/install-mongodb-on-windows/
from arctic import Arctic
from arctic import TICK_STORE
import pandas as pd

# Connect to local mongodb
store = Arctic('localhost')

# Import file from csv
file = "C:/Users/Administrator/Google Drive/Eiffeltower Capital/BaData/BA_EURUSD_15min.txt"
df = pd.read_csv(file, delimiter=',', index_col= 0, parse_dates=[[0,1]])

# Show head of file
df.head()

# Initialize the library
store.initialize_library('FX', lib_type = TICK_STORE)
library = store['FX']

# Write as EURUSD
library.write('EURUSD', df, metadata={'source':'BuildAlpha'})

# Read library EURUSD
item = library.read('EURUSD')

# Extract data
eurusd = item.data

# Extract metadata
metadata = item.metadata

# list symbols in library
library.list_symbols()

# get number of datapoints in storage
store.get_quota('FX')