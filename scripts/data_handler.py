import os
import logging
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Create log folder if it doesnot exist
log_directory = "../logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Configure logging
log_filename = os.path.join(log_directory, f'historical_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename)
    ]
)

class DataHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.scaler = StandardScaler()

    def load_data(self):
        """Loads CSV data, parses dates, and sets the date column as the index."""
        try:
            self.df = pd.read_csv(self.filepath, parse_dates=['Date'], index_col='Date')
            logging.info(f"Data loaded successfully from {self.filepath}")
        except Exception as e:
            logging.error(f"Error loading data from {self.filepath}: {e}")
            raise
        return self.df
