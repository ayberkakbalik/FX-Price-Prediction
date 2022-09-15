#Import the python libraries
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import copy

from pypfopt import plotting
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
#     from pypfopt import objective_functions

#Load the data
df = pd.read_csv('C://Users//AYBERK//OneDrive - ozyegin.edu.tr//Masaüstü//PortfolioData.csv')
#Set the date as the index
df = df.set_index(pd.DatetimeIndex(df['Date'].values))
#Remove the Date column
df.drop(columns=['Date'], axis=1, inplace=True)
print(df)

#Get the assets /tickers / Bizim için bu currencyler
assets = df.columns

#Calculate the expected annualized returns and the annualized sample covariance matrix of the daily asset returns
mu = expected_returns.capm_return(df,frequency=52) # frequency equals to 52 because of 52 weeks of a year
print(mu)
S = risk_models.exp_cov(df,frequency=52) # Recent data gain more weight thanks to exp_cov
print(S)

#Optimize for the maximal Sharpe ratio
ef = EfficientFrontier(mu, S) #Create the Efficient Frontier Object
fig, ax = plt.subplots()
ef_max_sharpe = copy.deepcopy(ef)
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
weights = ef_max_sharpe.max_sharpe(risk_free_rate = 0.0015) # It is determined United States treasury bonds rate
#cleaned_weights = ef.clean_weights()
print(weights)

ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance(verbose=True, risk_free_rate = 0.0015)
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

# Generate random portfolios
#n_samples = 10000
#w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
#rets = w.dot(ef.expected_returns)
#stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
#sharpes = rets / stds
#ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

#Get the discrete allocation of each share per stock
portfolio_val = 5000   #This will contain how much money USD that we want to put in this portfolio
latest_prices = get_latest_prices(df)
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = portfolio_val)
allocation, leftover = da.lp_portfolio()
print('Discrete allocation:', allocation)
print('Funds Remaining: $', leftover)

# Output
ax.set_title("Efficient Frontier of StemGNN")
ax.legend()
plt.tight_layout()
plt.savefig("ef_scatter.png", dpi=200)
plt.show()
