import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'V', 'JPM', 'JNJ', 
    'NVDA', 'PG', 'DIS', 'NFLX', 'XOM', 'KO', 'PEP', 'WMT', 'PFE', 'MA'
]

#Set up dictionaries 
stock_data_dict = {}
stats_dict = {}

for ticker in tickers: 
    stock_data = yf.Ticker(ticker).history(period="max")
    #Only need open and close prices 
    stock_data_dict[(ticker, 'Open')] = stock_data['Open']
    stock_data_dict[(ticker, 'Close')] = stock_data['Close']

    #Daily returns
    daily_returns = stock_data['Close'].pct_change().dropna()
    mean_return = daily_returns.mean()
    sigma = daily_returns.std()
    #Hold values in dictionary 
    stats_dict[ticker] = {
        'Mean Return': mean_return, 
        'Sigma': sigma,
        'Last Close': stock_data['Close'][-1]
    }

#Convert data to dataframe
all_stock_data = pd.DataFrame(stock_data_dict)
stats_df = pd.DataFrame(stats_dict).T

#print(all_stock_data)
#print(stats_df)

###### The Simulation ######
strating_balance = 40000
n_stocks = len(tickers)
allocation_per_stock = strating_balance / n_stocks
time_horizon = 60 #number of steps
n_simulations = 1000 #number of sims

#GBM
def simulate_gbm(S0, mu, sigma, T, N):
    dt = T / N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N) 
    W = np.cumsum(W) * np.sqrt(dt)  # Brownian motion
    X = (mu - 0.5 * sigma ** 2) * t + sigma * W
    S = S0 * np.exp(X)  # Geometric Brownian Motion
    return S
plt.figure(figsize=(10,6))

for sim in range(n_simulations):
    portfolio_values = np.zeros(time_horizon)

    #Running the simulations
    for ficker in tickers:
        S0 = stats_dict[ticker]['Last Close']
        mu = stats_dict[ticker]['Mean Return']
        sigma = stats_dict[ticker]['Sigma']
        n_shares = allocation_per_stock / S0 #shares per stock
        simulated_prices = simulate_gbm(S0, mu, sigma, time_horizon, time_horizon)
        stock_values = simulated_prices * n_shares #value at each step
        portfolio_values += stock_values 
    plt.plot(portfolio_values, label=f'Simulation {sim+1}', alpha = 0.3)

#Plotting
plt.title('Portfolio Value Simulation Over Time (30 Simulations)')
plt.xlabel('Time Steps')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.show()