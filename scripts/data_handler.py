import os
import logging
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
from IPython.display import display

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
    
    def process_data(self):
        try:
            logging.info("Starting data processing.")
            
            # Display the first few rows of the DataFrame
            display(self.df.head())
            logging.info("Displayed the first few rows of the DataFrame.")
            
            # Display the count of missing values
            missing_values = self.df.isnull().sum()
            display(missing_values)
            logging.info(f"Displayed missing values count:\n{missing_values}")
        
        except Exception as e:
            logging.error(f"Error processing data: {e}")
            raise

    def normalize_data(self):
        """Standardizes numerical columns in the data."""
        if self.df is not None:
            try:
                numerical_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                self.df[numerical_columns] = self.scaler.fit_transform(self.df[numerical_columns])
                logging.info("Data normalized successfully.")
            except KeyError as e:
                logging.error(f"Error in normalization, missing expected columns: {e}")
                raise
            except Exception as e:
                logging.error(f"Unexpected error during normalization: {e}")
                raise
        else:
            logging.warning("No data to normalize. Please load and clean data before attempting to normalize it.")
            raise ValueError("No data loaded.")
        return self.df
    


