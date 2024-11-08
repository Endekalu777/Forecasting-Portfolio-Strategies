import os
import logging
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