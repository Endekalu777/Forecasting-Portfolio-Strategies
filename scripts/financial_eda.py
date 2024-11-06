import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

class FinancialEDA:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.df['Date'] = pd.to_datetime(self.df['Date'])

    def plot_closing_prices(self, label):
        """Plots the closing prices over time."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.df['Date'], self.df['Close'], label=label)
        plt.title(f'{label} Closing Prices Over Time')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.show()

    def daily_percentage_change(self):
        """Calculates and plots the daily percentage change."""
        self.df['Daily Change'] = self.df['Close'].pct_change()
        plt.figure(figsize=(12, 6))
        plt.plot(self.df['Date'], self.df['Daily Change'], label='Daily Percentage Change')
        plt.title('Daily Percentage Change')
        plt.xlabel('Date')
        plt.ylabel('Percentage Change')
        plt.legend()
        plt.show()

    def rolling_stats(self, window=30):
        """Calculates and plots rolling mean and standard deviation."""
        self.df['Rolling Mean'] = self.df['Close'].rolling(window=window).mean()
        self.df['Rolling Std'] = self.df['Close'].rolling(window=window).std()
        plt.figure(figsize=(12, 6))
        plt.plot(self.df['Close'], label='Close Price')
        plt.plot(self.df['Rolling Mean'], label=f'{window}-Day Rolling Mean')
        plt.fill_between(self.df.index, self.df['Rolling Mean'] - self.df['Rolling Std'], 
                         self.df['Rolling Mean'] + self.df['Rolling Std'], color='b', alpha=0.2)
        plt.title(f'Rolling Mean and Standard Deviation ({window}-day)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def trend_seasonality_decomposition(self):
        """Performs time series decomposition into trend, seasonality, and residuals."""
        result = seasonal_decompose(self.df['Close'], model='multiplicative', period=365)
        result.plot()
        plt.show()

    def risk_metrics(self):
        """Calculates Value at Risk (VaR) and Sharpe Ratio."""
        daily_change = self.df['Daily Change'].dropna()
        
        # Calculate 95% VaR
        var_95 = np.percentile(daily_change, 5)
        print(f"95% Value at Risk (VaR): {var_95}")

        # Calculate Sharpe Ratio (assuming risk-free rate is 0)
        sharpe_ratio = daily_change.mean() / daily_change.std()
        print(f"Sharpe Ratio: {sharpe_ratio}")