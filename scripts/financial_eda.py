import matplotlib.pyplot as plt
import pandas as pd

class FinancialEDA:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

    def plot_closing_prices(self, label):
        """Plots the closing prices over time."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.df['Close'], label=label)
        plt.title(f'{label} Closing Prices Over Time')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.show()