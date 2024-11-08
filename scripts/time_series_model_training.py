import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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

    def train_lstm(self, seq_length=60, epochs=50, batch_size=32):
        """Train LSTM model and save it"""
        try:
            logging.info("Training LSTM model")
            # Remove the scaling step
            data = self.df.values
            
            # Split data
            train_size = len(self.train)
            train_data = data[:train_size]
            
            # Create sequences
            X_train, y_train = self.create_sequences(train_data, seq_length)
            
            # Build LSTM model
            model = Sequential([ 
                LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
                Dropout(0.2),
                LSTM(50, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            
            # Train model
            history = model.fit(X_train, y_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_split=0.1,
                              verbose=1)
            
            # Save only the LSTM model
            model_save_path = f"../models/lstm_model_{self.filepath.split('/')[-1].split('.')[0]}.h5"
            model.save(model_save_path)
            logging.info(f"LSTM model saved to {model_save_path}")
            
            self.models['LSTM'] = {
                'model': model,
                'seq_length': seq_length,
                'history': history
            }
            logging.info("LSTM model trained successfully")
        except Exception as e:
            logging.error(f"Error in LSTM training: {e}")
            raise

    def make_predictions(self):
        """Make predictions using all trained models"""
        try:
            # ARIMA predictions
            if 'ARIMA' in self.models:
                self.predictions['ARIMA'] = self.models['ARIMA'].predict(n_periods=len(self.test))
                
            # SARIMA predictions
            if 'SARIMA' in self.models:
                self.predictions['SARIMA'] = self.models['SARIMA'].predict(n_periods=len(self.test))
                
            # LSTM predictions
            if 'LSTM' in self.models:
                seq_length = self.models['LSTM']['seq_length']
                model = self.models['LSTM']['model']
                
                # Prepare data for LSTM predictions
                data = self.df.values
                X_test = []
                for i in range(len(self.test)):
                    end_idx = len(self.train) + i
                    sequence = data[end_idx - seq_length:end_idx].reshape(-1)
                    X_test.append(sequence)
                X_test = np.array(X_test).reshape(-1, seq_length, 1)
                
                # Make predictions
                lstm_predictions = model.predict(X_test)
                self.predictions['LSTM'] = lstm_predictions
                
            logging.info("Predictions generated for all models")
        except Exception as e:
            logging.error(f"Error in making predictions: {e}")
            raise

    def evaluate_models(self):
        for model_name, model in self.models.items():
            logging.info(f"Evaluating model: {model_name}")

            # Generate predictions
            if model_name in ['ARIMA', 'SARIMA']:
                predictions = model.predict(n_periods=len(self.test))
                # Handle Pandas Series/DataFrame or NumPy array
                if isinstance(predictions, (pd.Series, pd.DataFrame)):
                    predictions = predictions.values.flatten()
                else:
                    predictions = predictions.flatten()
            elif model_name == 'LSTM':
                # Prepare data for LSTM predictions
                seq_length = self.models['LSTM']['seq_length']
                data = self.df.values
                X_test = []
                for i in range(len(self.test)):
                    end_idx = len(self.train) + i
                    sequence = data[end_idx - seq_length:end_idx].reshape(-1)
                    X_test.append(sequence)
                X_test = np.array(X_test).reshape(-1, seq_length, 1)
                predictions = model['model'].predict(X_test)
                predictions = predictions.flatten()

            test_data = np.array(self.test).flatten()

            print(f"Shape of test_data for {model_name}:", test_data.shape)
            print(f"Shape of predictions for {model_name}:", predictions.shape)

            mae = mean_absolute_error(test_data, predictions)
            rmse = np.sqrt(mean_squared_error(test_data, predictions))
            mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100

            self.metrics[model_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            }

            logging.info(f"{model_name} Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

    def plot_results(self):
        """Plot results for all models"""
        try:
            plt.figure(figsize=(15, 8))
            plt.plot(self.test.index, self.test.values, label='Actual', linewidth=2)
            
            colors = ['red', 'green', 'blue']
            for (model_name, predictions), color in zip(self.predictions.items(), colors):
                plt.plot(self.test.index, predictions, label=f'{model_name} Predictions',
                        linestyle='--', color=color)
            
            plt.title('Model Predictions Comparison')
            plt.xlabel('Date')
            plt.ylabel(self.column)
            plt.legend()
            plt.show()
            
            logging.info("Results plotted successfully")
        except Exception as e:
            logging.error(f"Error in plotting results: {e}")
            raise
    
    def forecast_future(self, periods=30):
        """Forecast future values using the LSTM model only"""
        try:
            forecasts = {}

            if 'LSTM' in self.models:
                # Implementation for LSTM future forecasting
                model = self.models['LSTM']['model']
                seq_length = self.models['LSTM']['seq_length']
                
                # Prepare data for future forecasting (use the last seq_length data points)
                data = self.df.values
                last_sequence = data[-seq_length:].reshape(1, seq_length, 1)
                
                # Predict future values sequentially
                future_predictions = []
                for _ in range(periods):
                    prediction = model.predict(last_sequence)
                    future_predictions.append(prediction[0, 0])
                    
                    # Update the last_sequence with the predicted value
                    last_sequence = np.append(last_sequence[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

                forecasts['LSTM'] = future_predictions
                
                logging.info(f"Generated forecasts for {periods} periods using LSTM")
            else:
                logging.error("LSTM model is not trained yet")
                raise ValueError("LSTM model is not available for forecasting")

            return forecasts
        except Exception as e:
            logging.error(f"Error in forecasting: {e}")
            raise
