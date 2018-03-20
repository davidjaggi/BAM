# https://gist.github.com/TeamAuquan/b90b348caf4573ebf44a5751ef633b7d#file-apply-ml-to-trading-ipynb
# Import packages

 
import numpy
import scipy
import sklearn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

# Get FX prices
fx = "C:/Users/Administrator/Desktop/BaData/BA_GBPUSD_15min.txt"