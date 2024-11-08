import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
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
    handlers=[logging.FileHandler(log_filename)]
)

class TimeSeriesForecaster:
    def __init__(self, filepath, column='Close'):
        self.filepath = filepath
        self.column = column
        self.df = self._load_data()
        self.train = None
        self.test = None
        self.models = {}
        self.predictions = {}
        self.metrics = {}

    def _load_data(self):
        """Loads data from CSV file."""
        try:
            df = pd.read_csv(self.filepath)
            logging.info(f"Data loaded successfully from {self.filepath}")
            return df[self.column]
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise

    def split_data(self, train_size=0.8):
        """Splits data into training and test sets."""
        try:
            split_index = int(len(self.df) * train_size)
            self.train = self.df[:split_index]
            self.test = self.df[split_index:]
            logging.info(f"Data split into train ({len(self.train)}) and test ({len(self.test)}) sets")
        except Exception as e:
            logging.error(f"Error in data splitting: {e}")
            raise

    def create_sequences(self, data, seq_length):
        """Create sequences for LSTM model"""
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            xs.append(data[i:(i + seq_length)])
            ys.append(data[i + seq_length])
        return np.array(xs), np.array(ys)
    
    def train_arima(self):
        """Train ARIMA model"""
        try:
            logging.info("Training ARIMA model")
            model = auto_arima(self.train, seasonal=False,
                             start_p=0, start_q=0,
                             max_p=5, max_q=5,
                             d=1, trace=True,
                             error_action='ignore',
                             suppress_warnings=True)
            
            self.models['ARIMA'] = model
            logging.info(f"ARIMA model trained with parameters: {model.get_params()}")
        except Exception as e:
            logging.error(f"Error in ARIMA training: {e}")
            raise

    def train_sarima(self, seasonal_period=5):
        """Train SARIMA model"""
        try:
            logging.info("Training SARIMA model")
            model = auto_arima(self.train,
                             seasonal=True,
                             m=seasonal_period,
                             start_p=0, start_q=0,
                             max_p=3, max_q=3,
                             start_P=0, start_Q=0,
                             max_P=2, max_Q=2,
                             d=1, D=1,
                             trace=True,
                             error_action='ignore',
                             suppress_warnings=True)
            
            self.models['SARIMA'] = model
            logging.info(f"SARIMA model trained with parameters: {model.get_params()}")
        except Exception as e:
            logging.error(f"Error in SARIMA training: {e}")
            raise