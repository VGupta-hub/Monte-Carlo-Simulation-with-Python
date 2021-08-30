# Importing the libraries
import numpy as np
import pandas as pd
from pandas_datareader import data as web
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.stats import norm
%matplotlib notebook

# Pulling the data for AAPL (I have used a CSV file and data from 01/01/2021 - 08/27/2021)
df = pd.read_csv("C:/Users/vgupt/Desktop/Website Project Folder/MCS/AAPL.csv")['Adj Close']
df

#Ploting the prices on a graph
plt.xlabel("Date")
plt.ylabel("Price")
df.plot(figsize=(15,6))

# Calulating the logarithmic returns returns of AAPL stock
log_return = np.log(1 + df.pct_change())
log_return

# Computing the drift
u = log_return.mean()
v = log_return.var()

drift = u - (0.5*v)
drift

# Computing the variance and Daily Returns
stdev = log_return.std()
days = 50  #We are predicting AAPL's stock prices 50 days into the future
num_simulations = 1000
Z = norm.ppf(np.random.rand(days,num_simulations))
Z

daily_returns = np.exp(drift + stdev*Z)

plt.plot(daily_returns)

# Calulating the stock price for every trial
price_paths = np.zeros_like(daily_returns)
price_paths[0] = df.iloc[-1]
for t in range(1, days):
    price_paths[t] = price_paths[t-1]*daily_returns[t]
    
price_paths

plt.plot(price_paths)



