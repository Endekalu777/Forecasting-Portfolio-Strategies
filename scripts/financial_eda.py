import matplotlib.pyplot as plt
import pandas as pd

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