# https://gist.github.com/TeamAuquan/b90b348caf4573ebf44a5751ef633b7d#file-apply-ml-to-trading-ipynb
# Import packages

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

# Get FX prices
fx_fileName = "C:/Users/David Jaggi/Google Drive/Eiffeltower Capital/BaData/BA_GBPUSD_15min.txt"
data = pd.read_csv(fx_fileName, index_col = [[0,1]], parse_dates=[[0,1]])
data = pd.read
head(data)
data.plot()