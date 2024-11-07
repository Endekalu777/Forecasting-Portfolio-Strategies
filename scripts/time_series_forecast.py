import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
import os
import logging
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
    handlers=[logging.FileHandler(log_filename)]
)

class TimeSeriesForecaster:
    def __init__(self, filepath, column='Close'):
        self.filepath = filepath
        self.column = column
        self.df = self._load_data()
        self.train, self.test = None, None
        self.best_params = None
        self.model = None
        self.fitted_model = None
        self.predictions = None
        self.metrics = None

    def _load_data(self):
        """Loads data from CSV file."""
        try:
            df = pd.read_csv(self.filepath)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            logging.info(f"Data loaded successfully from {self.filepath}")
            return df[self.column]
        except Exception as e:
            logging.error(f"Failed to load data from {self.filepath}: {e}")
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

    def optimize_parameters(self, seasonal_period=5):
        """Optimizes SARIMA parameters using auto_arima."""
        try:
            logging.info("Starting SARIMA parameter optimization with auto_arima")
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
                               suppress_warnings=True,
                               stepwise=True)
            self.best_params = model.get_params()
            logging.info(f"Best parameters found: {self.best_params}")
        except Exception as e:
            logging.error(f"Error in SARIMA parameter optimization: {e}")
            raise

    def train_model(self):
        """Trains SARIMA model with optimized parameters."""
        try:
            if not self.best_params:
                raise ValueError("Parameters not set. Run optimize_parameters() first.")
            order = self.best_params['order']
            seasonal_order = self.best_params['seasonal_order']
            self.model = SARIMAX(self.train, order=order, seasonal_order=seasonal_order)
            self.fitted_model = self.model.fit(disp=False)
            logging.info("Model training completed successfully")
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise

    def make_predictions(self):
        """Makes predictions on test data."""
        try:
            self.predictions = self.fitted_model.forecast(steps=len(self.test))
            logging.info(f"Generated {len(self.test)} predictions")
        except Exception as e:
            logging.error(f"Error in making predictions: {e}")
            raise

    def evaluate_model(self):
        """Evaluates model performance using MAE, RMSE, and MAPE metrics."""
        try:
            mae = mean_absolute_error(self.test, self.predictions)
            rmse = np.sqrt(mean_squared_error(self.test, self.predictions))
            mape = np.mean(np.abs((self.test - self.predictions) / self.test)) * 100
            self.metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
            logging.info(f"Model evaluation completed: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
            return self.metrics
        except Exception as e:
            logging.error(f"Error in model evaluation: {e}")
            raise

    def plot_results(self):
        """Plots actual vs predicted values."""
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(self.test.index, self.test.values, label='Actual')
            plt.plot(self.test.index, self.predictions, label='Predicted')
            plt.title('SARIMA Model - Actual vs Predicted')
            plt.xlabel('Date')
            plt.ylabel(self.column)
            plt.legend()
            plt.show()
            logging.info("Results plotted successfully")
        except Exception as e:
            logging.error(f"Error in plotting results: {e}")
            raise

    def forecast_future(self, periods=30):
        """Forecasts future values."""
        try:
            future_forecast = self.fitted_model.get_forecast(steps=periods).predicted_mean
            logging.info(f"Forecasted future values for {periods} periods")
            return future_forecast
        except Exception as e:
            logging.error(f"Error in forecasting future values: {e}")
            raise