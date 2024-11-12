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