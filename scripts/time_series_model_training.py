import os
import logging
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Setup logging
log_directory = "../logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

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
        self.data_df = pd.read_csv(self.filepath)
        self.train = None
        self.test = None
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Scaler for LSTM and other models

    def _load_data(self):
        """Loads data from CSV file."""
        try:
            df = pd.read_csv(self.filepath)
            logging.info(f"Data loaded successfully from {self.filepath}")
            return df[[self.column]]  # Return DataFrame instead of Series
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise

    def scale_data(self, scaler_save_path):
        """Scale data using MinMaxScaler and save the scaler."""
        try:
            logging.info("Scaling data for all models")
            self.df[self.column] = self.scaler.fit_transform(self.df[[self.column]])

            # Save the fitted scaler
            joblib.dump(self.scaler, scaler_save_path)
            logging.info(f"Scaler saved to {scaler_save_path}")

        except Exception as e:
            logging.error(f"Error in data scaling: {e}")
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
        try:
            if isinstance(data, pd.DataFrame):
                values = data[self.column].values
            elif isinstance(data, pd.Series):
                values = data.values
            else:
                values = np.array(data)
            
            xs, ys = [], []
            for i in range(len(values) - seq_length):
                xs.append(values[i:(i + seq_length)])
                ys.append(values[i + seq_length])
            
            return np.array(xs), np.array(ys)
        except Exception as e:
            logging.error(f"Error in creating sequences: {e}")
            raise

    def train_arima(self):
        """Train ARIMA model"""
        try:
            logging.info("Training ARIMA model")
            model = auto_arima(self.train, seasonal=False, start_p=0, start_q=0, max_p=5, max_q=5, d=1, trace=True, error_action='ignore', suppress_warnings=True)
            self.models['ARIMA'] = model
            logging.info(f"ARIMA model trained with parameters: {model.get_params()}")
        except Exception as e:
            logging.error(f"Error in ARIMA training: {e}")
            raise

    def train_sarima(self, seasonal_period=5):
        """Train SARIMA model"""
        try:
            logging.info("Training SARIMA model")
            model = auto_arima(self.train, seasonal=True, m=seasonal_period, start_p=0, start_q=0, max_p=3, max_q=3, start_P=0, start_Q=0, max_P=2, max_Q=2, d=1, D=1, trace=True, error_action='ignore', suppress_warnings=True)
            self.models['SARIMA'] = model
            logging.info(f"SARIMA model trained with parameters: {model.get_params()}")
        except Exception as e:
            logging.error(f"Error in SARIMA training: {e}")
            raise

    def train_lstm(self, seq_length=60, epochs=50, batch_size=32):
        """Train LSTM model and save it"""
        try:
            logging.info("Training LSTM model")
            data = self.df.values
            train_data = data[:len(self.train)]
            X_train, y_train = self.create_sequences(train_data, seq_length)

            model = Sequential([ 
                LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
                Dropout(0.2),
                LSTM(50, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse')
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

            model_save_path = f"../models/lstm_model_{self.filepath.split('/')[-1].split('.')[0]}.h5"
            model.save(model_save_path)
            logging.info(f"LSTM model saved to {model_save_path}")

            self.models['LSTM'] = {'model': model, 'seq_length': seq_length, 'history': history}
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
                data = self.df.values
                X_test = []
                for i in range(len(self.test)):
                    end_idx = len(self.train) + i
                    sequence = data[end_idx - seq_length:end_idx].reshape(-1)
                    X_test.append(sequence)
                X_test = np.array(X_test).reshape(-1, seq_length, 1)

                lstm_predictions = model.predict(X_test)
                self.predictions['LSTM'] = lstm_predictions

            logging.info("Predictions generated for all models")
        except Exception as e:
            logging.error(f"Error in making predictions: {e}")
            raise

    def evaluate_models(self):
        for model_name, model in self.models.items():
            logging.info(f"Evaluating model: {model_name}")

            if model_name in ['ARIMA', 'SARIMA']:
                predictions = model.predict(n_periods=len(self.test))
                if isinstance(predictions, (pd.Series, pd.DataFrame)):
                    predictions = predictions.values.flatten()
                else:
                    predictions = predictions.flatten()
            elif model_name == 'LSTM':
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

            mae = mean_absolute_error(test_data, predictions)
            rmse = np.sqrt(mean_squared_error(test_data, predictions))
            mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100

            self.metrics[model_name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
            logging.info(f"{model_name} Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

    def plot_results(self):
        """Plot results for all models"""
        # Ensure 'Date' is in datetime format
        self.data_df['Date'] = pd.to_datetime(self.data_df['Date'])

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot the actual data (test data) using the Date column from data_df
        plt.plot(self.data_df['Date'].iloc[-len(self.test):], self.test, label='Actual Data')
        
        # Plot predictions from all models
        for model_name, predictions in self.predictions.items():
            # Plot predictions using the 'Date' column from data_df
            plt.plot(self.data_df['Date'].iloc[-len(self.test):], predictions, label=f'{model_name} Predictions')

        # Set plot title and labels
        plt.title("Model Predictions vs Actual Data")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()

        # Show plot
        plt.show()
    
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
