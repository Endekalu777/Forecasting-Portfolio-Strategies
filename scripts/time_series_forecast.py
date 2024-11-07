import logging
import os
from datetime import datetime
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

# Create log folder if it does not exist
log_directory = "../logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Configure logging
log_filename = os.path.join(log_directory, f'time_series_forecast_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename)
    ]
)

class TimeSeriesForecaster:
    def __init__(self, filepath):
        try:
            self.df = pd.read_csv(filepath)
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df.set_index('Date', inplace=True)
            self.df = self.df['Close']  
            logging.info(f"Data successfully loaded from {filepath}")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def split_data(self, train_size=0.8):
        """Split the data into training and testing sets."""
        try:
            train_size = int(len(self.df) * train_size)
            self.train = self.df[:train_size]
            self.test = self.df[train_size:]
            logging.info(f"Data split into train ({len(self.train)} samples) and test ({len(self.test)} samples)")
            return self.train, self.test
        except Exception as e:
            logging.error(f"Error splitting data: {e}")
            raise

    def find_best_parameters(self):
        """Find the best SARIMA parameters using auto_arima."""
        try:
            logging.info("Starting parameter optimization with auto_arima")
            model = auto_arima(self.train,
                             seasonal=True,
                             m=5,  # Weekly seasonality
                             start_p=0, start_q=0,
                             max_p=3, max_q=3,
                             start_P=0, start_Q=0,
                             max_P=2, max_Q=2,
                             d=1, D=1,
                             trace=True,
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=True)
            
            self.best_params = model.get_params()
            logging.info(f"Best parameters found: {self.best_params}")
            return self.best_params
        except Exception as e:
            logging.error(f"Error in parameter optimization: {e}")
            raise

    def train_model(self):
        """Train the SARIMA model with the best parameters."""
        try:
            order = self.best_params['order']
            seasonal_order = self.best_params['seasonal_order']
            
            self.model = SARIMAX(self.train,
                                order=order,
                                seasonal_order=seasonal_order)
            
            self.fitted_model = self.model.fit()
            logging.info("Model training completed successfully")
        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise
    