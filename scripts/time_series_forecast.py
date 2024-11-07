import logging
import os
from datetime import datetime

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
            self.df = self.df['Close']  # Using only closing prices
            logging.info(f"Data successfully loaded from {filepath}")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise