import os
import logging 
from datetime import datetime
import numpy as np

# Setup logging
log_directory = "../logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_filename = os.path.join(log_directory, f'portfolio_optimizer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_filename)]
)

class PortfolioOptimizer:
    def __init__(self, forecast_df_tsla, forecast_df_bnd, forecast_df_spy):
        logging.info("Initializing PortfolioOptimizer with provided forecast data.")
        self.forecast_df = self.combine_forecasts(forecast_df_tsla, forecast_df_bnd, forecast_df_spy)
        self.annual_trading_days = 252 
        self.initial_weights = np.array([0.33, 0.33, 0.34])  
        self.risk_free_rate = 0.03  
        logging.info("PortfolioOptimizer initialized.")

    def combine_forecasts(self, tsla_df, bnd_df, spy_df):
        """Combine forecast dataframes into one with columns TSLA, BND, SPY."""
        logging.info("Combining forecast dataframes for TSLA, BND, and SPY.")
        combined_df = pd.DataFrame({
            'Date': tsla_df['Date'],
            'TSLA': tsla_df['Forecast'],
            'BND': bnd_df['Forecast'],
            'SPY': spy_df['Forecast']
        })
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        combined_df.set_index('Date', inplace=True)
        logging.info("Forecast data combined successfully.")
        return combined_df
    
    def calculate_daily_returns(self):
        """Calculate daily returns for each asset in the forecast dataframe."""
        logging.info("Calculating daily returns.")
        daily_returns = self.forecast_df.pct_change().dropna()
        logging.info("Daily returns calculated.")
        return daily_returns