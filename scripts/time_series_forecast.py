import logging
import os
from datetime import datetime
import pandas as pd

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

    