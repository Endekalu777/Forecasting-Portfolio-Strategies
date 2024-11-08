import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np

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