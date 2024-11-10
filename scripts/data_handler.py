import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import joblib
import warnings
warnings.filterwarnings('ignore')

# Create log folder if it doesn't exist
log_directory = "../logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Configure logging
log_filename = os.path.join(log_directory, f'historical_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
    ]
)

class DataHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.scaler = RobustScaler()
        self.numerical_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    def load_data(self):
        """Loads CSV data, parses dates, and sets the date column as the index."""
        try:
            self.df = pd.read_csv(self.filepath, parse_dates=['Date'], index_col='Date')
            logging.info(f"Data loaded successfully from {self.filepath}")
        except Exception as e:
            logging.error(f"Error loading data from {self.filepath}: {e}")
            raise
        return self.df

    def detect_outliers(self, contamination=0.01):
        """Detect outliers using Isolation Forest."""
        try:
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outliers = iso_forest.fit_predict(self.df[self.numerical_columns])
            self.df['is_outlier'] = outliers
            outlier_count = len(self.df[self.df['is_outlier'] == -1])
            
            # Print the number of outliers
            print(f"Number of detected outliers: {outlier_count}")
            
            logging.info(f"Detected {outlier_count} outliers")
        except Exception as e:
            logging.error(f"Error detecting outliers: {e}")
            raise

    def handle_outliers(self, method='clip'):
        """Handle outliers using specified method."""
        try:
            if method == 'clip':
                for column in self.numerical_columns:
                    Q1 = self.df[column].quantile(0.25)
                    Q3 = self.df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    self.df[column] = self.df[column].clip(lower_bound, upper_bound)
            elif method == 'remove':
                self.df = self.df[self.df['is_outlier'] == 1]
            logging.info(f"Outliers handled using {method} method")
        except Exception as e:
            logging.error(f"Error handling outliers: {e}")
            raise

    def plot_outliers(self):
        """Plot outliers for visualization."""
        try:
            plt.figure(figsize=(15, 10))
            for i, column in enumerate(self.numerical_columns, 1):
                plt.subplot(2, 3, i)
                plt.scatter(self.df.index, self.df[column], 
                          c=self.df['is_outlier'], cmap='viridis')
                plt.title(f'Outliers in {column}')
                plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            logging.info("Outlier plots generated successfully")
        except Exception as e:
            logging.error(f"Error plotting outliers: {e}")
            raise

    def process_data(self):
        try:
            logging.info("Starting data processing.")
            
            # Basic statistics
            stats = self.df.describe()
            display(stats)
            logging.info("Displayed basic statistics")
            
            # Missing values
            missing_values = self.df.isnull().sum()
            display(missing_values)
            logging.info(f"Missing values count:\n{missing_values}")
            
            # Handle missing values
            self.df.fillna(method='ffill', inplace=True)
            self.df.fillna(method='bfill', inplace=True)
            
            # Detect and handle outliers
            self.detect_outliers()
            self.handle_outliers(method='clip')
            self.plot_outliers()
            
        except Exception as e:
            logging.error(f"Error processing data: {e}")
            raise

    def normalize_data(self):
        """Standardizes numerical columns in the data."""
        if self.df is not None:
            try:
                self.df[self.numerical_columns] = self.scaler.fit_transform(self.df[self.numerical_columns])
                logging.info("Data normalized successfully.")
            except Exception as e:
                logging.error(f"Error during normalization: {e}")
                raise
        else:
            logging.warning("No data to normalize.")
            raise ValueError("No data loaded.")
        return self.df

    def save_scaler(self, scaler_filename):
        """Saves the fitted scaler to a file."""
        try:
            joblib.dump(self.scaler, scaler_filename)
            logging.info(f"Scaler saved successfully to {scaler_filename}")
        except Exception as e:
            logging.error(f"Error saving scaler: {e}")
            raise

    def save_processed_data(self, output_filepath):
        """Save processed data to CSV."""
        try:
            # Remove the is_outlier column before saving
            if 'is_outlier' in self.df.columns:
                self.df = self.df.drop('is_outlier', axis=1)
            
            # Reset index to include Date as a column
            df_to_save = self.df.reset_index()
            
            # Save to CSV with Date column
            df_to_save.to_csv(output_filepath, index=False)
            logging.info(f"Processed data saved to {output_filepath}")
        except Exception as e:
            logging.error(f"Error saving processed data: {e}")
            raise