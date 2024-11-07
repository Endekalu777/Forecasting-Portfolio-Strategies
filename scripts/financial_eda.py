import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import os
import logging
from datetime import datetime

# Create log folder if it does not exist
log_directory = "../logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Configure logging
log_filename = os.path.join(log_directory, f'financial_eda_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename)
    ]
)

class FinancialEDA:
    def __init__(self, filepath):
        try:
            self.df = pd.read_csv(filepath)
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            logging.info(f"Data successfully loaded from {filepath}")
        except Exception as e:
            logging.error(f"Error loading data from {filepath}: {e}")
            raise

    def plot_closing_prices(self, label):
        """Plots the closing prices over time."""
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(self.df['Date'], self.df['Close'], label=label)
            plt.title(f'{label} Closing Prices Over Time')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.legend()
            plt.show()
            logging.info(f"Successfully plotted closing prices for {label}")
        except Exception as e:
            logging.error(f"Error plotting closing prices: {e}")
            raise

    def daily_percentage_change(self):
        """Calculates and plots the daily percentage change."""
        try:
            self.df['Daily Change'] = self.df['Close'].pct_change()
            plt.figure(figsize=(12, 6))
            plt.plot(self.df['Date'], self.df['Daily Change'], label='Daily Percentage Change')
            plt.title('Daily Percentage Change')
            plt.xlabel('Date')
            plt.ylabel('Percentage Change')
            plt.legend()
            plt.show()
            logging.info("Successfully calculated and plotted daily percentage change")
        except Exception as e:
            logging.error(f"Error calculating daily percentage change: {e}")
            raise

    def rolling_stats(self, window=30):
        """Calculates and plots rolling mean and standard deviation."""
        try:
            self.df['Rolling Mean'] = self.df['Close'].rolling(window=window).mean()
            self.df['Rolling Std'] = self.df['Close'].rolling(window=window).std()
            plt.figure(figsize=(12, 6))
            plt.plot(self.df['Close'], label='Close Price')
            plt.plot(self.df['Rolling Mean'], label=f'{window}-Day Rolling Mean')
            plt.fill_between(self.df.index, 
                           self.df['Rolling Mean'] - self.df['Rolling Std'], 
                           self.df['Rolling Mean'] + self.df['Rolling Std'], 
                           color='b', alpha=0.2)
            plt.title(f'Rolling Mean and Standard Deviation ({window}-day)')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.show()
            logging.info(f"Successfully calculated and plotted rolling statistics with {window}-day window")
        except Exception as e:
            logging.error(f"Error calculating rolling statistics: {e}")
            raise

    def trend_seasonality_decomposition(self):
        """Performs time series decomposition into trend, seasonality, and residuals."""
        try:
            result = seasonal_decompose(self.df['Close'], model='multiplicative', period=365)
            result.plot()
            plt.show()
            logging.info("Successfully performed trend seasonality decomposition")
        except Exception as e:
            logging.error(f"Error performing trend seasonality decomposition: {e}")
            raise

    def risk_metrics(self):
        """Calculates Value at Risk (VaR) and Sharpe Ratio."""
        try:
            daily_change = self.df['Daily Change'].dropna()
            
            # Calculate 95% VaR
            var_95 = np.percentile(daily_change, 5)
            logging.info(f"Calculated 95% Value at Risk (VaR): {var_95}")
            print(f"95% Value at Risk (VaR): {var_95}")

            # Calculate Sharpe Ratio (assuming risk-free rate is 0)
            sharpe_ratio = daily_change.mean() / daily_change.std()
            logging.info(f"Calculated Sharpe Ratio: {sharpe_ratio}")
            print(f"Sharpe Ratio: {sharpe_ratio}")
            
        except Exception as e:
            logging.error(f"Error calculating risk metrics: {e}")
            raise