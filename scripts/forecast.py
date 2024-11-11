import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy import stats
from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.models import load_model

class MarketTrendAnalyzer:
    def __init__(self, model_path, scaler_path, historical_data_path):
        self.model = self.load_model(model_path)
        self.scaler = self.load_scaler(scaler_path)
        self.historical_data = self.load_historical_data(historical_data_path)
        self.forecast_periods = 180
        self.confidence_level = 0.95
        self.seq_length = 90
        
    def load_model(self, model_path):
        """Load and return the pre-trained model."""
        try:
            custom_objects = {'mse': mse}
            model = load_model(model_path, custom_objects=custom_objects)
            print(f"LSTM model successfully loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            raise
    
    def load_scaler(self, scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print(f"Scaler successfully loaded from {scaler_path}")
            return scaler
        except Exception as e:
            print(f"Error loading scaler from {scaler_path}: {e}")
            raise
    
    def load_historical_data(self, historical_data_path):
        try:
            data = pd.read_csv(historical_data_path)
            data['Date'] = pd.to_datetime(data['Date'])
            return data
        except Exception as e:
            print(f"Error loading historical data from {historical_data_path}: {e}")
            raise

    def generate_forecast(self, save_path=None):
        # Scale the input data
        scaled_data = self.scaler.transform(self.historical_data[['Close']]).flatten()
        
        # Prepare initial sequence
        last_sequence = scaled_data[-self.seq_length:].reshape(-1)
        scaled_forecast = []
        
        # Generate forecasted values
        current_sequence = last_sequence.copy()
        for t in range(self.forecast_periods):
            input_seq = current_sequence.reshape(1, self.seq_length, 1)
            pred = self.model.predict(input_seq, verbose=0)[0, 0]
            scaled_forecast.append(pred)
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred
        
        # Reshape and inverse transform the forecasted values
        scaled_forecast = np.array(scaled_forecast).reshape(-1, 1)
        forecasted_prices = self.scaler.inverse_transform(scaled_forecast).flatten()
        
        # Calculate confidence intervals with time-decay factor
        last_known_price = self.historical_data['Close'].iloc[-1]
        forecast_std = np.std(self.historical_data['Close'].pct_change().dropna())
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        
        # Time decay factor for increasing uncertainty
        time_decay = np.linspace(1, 1.5, num=self.forecast_periods)
        confidence_intervals = {
            'lower': forecasted_prices - (z_score * forecast_std * forecasted_prices * time_decay),
            'upper': forecasted_prices + (z_score * forecast_std * forecasted_prices * time_decay)
        }

        forecast_dates = pd.date_range(start=self.historical_data['Date'].max() + timedelta(days=1), 
                                    periods=self.forecast_periods, freq='D')

        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecasted_prices,
            'Lower_CI': confidence_intervals['lower'],
            'Upper_CI': confidence_intervals['upper']
        })

        # Save the forecast to a CSV file if save_path is provided
        if save_path:
            forecast_df.to_csv(save_path, index=False)
            print(f"Forecast saved to {save_path}")

        return forecast_df

    def plot_forecast(self):
        """Plot historical data, forecast, and confidence intervals."""
        plt.figure(figsize=(15, 8))

        # Plot historical data (original Close prices)
        plt.plot(self.historical_data['Date'], self.historical_data['Close'], label='Historical Data', color='blue')
        
        # Plot forecast
        forecast_df = self.generate_forecast()
        plt.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', color='red', linestyle='--')
        
        # Plot confidence intervals
        plt.fill_between(forecast_df['Date'], forecast_df['Lower_CI'], forecast_df['Upper_CI'],
                         alpha=0.2, color='red', label=f'{self.confidence_level*100}% Confidence Interval')
        
        plt.title('Stock Price Forecast with Confidence Intervals')
        plt.xlabel('Date')
        plt.ylabel('Stock Price ($)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def analyze_trends(self):
        """Analyze market trends with trend magnitude capped at 15%."""
        forecast_df = self.generate_forecast()
        
        # Calculate percentage-based trend
        initial_price = forecast_df['Forecast'].iloc[0]
        final_price = forecast_df['Forecast'].iloc[-1]
        trend_pct = ((final_price - initial_price) / initial_price) * 100

        # Cap trend magnitude and calculate volatility
        forecast_volatility = min(forecast_df['Forecast'].pct_change().std(), 0.02)  # Cap at 2%

        return {
            'Trend': 'Upward' if trend_pct > 0 else 'Downward',
            'Trend_Magnitude': min(abs(trend_pct), 15),
            'Volatility': forecast_volatility,
            'Max_Price': forecast_df['Forecast'].max(),
            'Min_Price': forecast_df['Forecast'].min(),
            'Price_Range': forecast_df['Forecast'].max() - forecast_df['Forecast'].min()
        }
    
    def generate_report(self):
        """Generate a report based on trend analysis."""
        risk_metrics = self.analyze_trends()
        report = f"""
        Market Trend Analysis Report
        ============================

        1. Overall Trend Analysis:
        -------------------------
        Direction: {risk_metrics['Trend']}
        Trend Magnitude: {risk_metrics['Trend_Magnitude']:.2f}

        2. Price Projections:
        --------------------
        Forecasted Maximum Price: ${risk_metrics['Max_Price']:.2f}
        Forecasted Minimum Price: ${risk_metrics['Min_Price']:.2f}
        Expected Price Range: ${risk_metrics['Price_Range']:.2f}

        3. Volatility Analysis:
        ----------------------
        Forecasted Volatility: {risk_metrics['Volatility']*100:.2f}%

        4. Market Opportunities and Risks:
        -------------------------------
        Main Opportunities:
        * {'Price appreciation potential' if risk_metrics['Trend'] == 'Upward' else 'Potential buying opportunities during dips'}
        * {'Momentum trading opportunities' if risk_metrics['Volatility'] > 0.02 else 'Stable price movement expected'}

        Main Risks:
        * {'High volatility risk' if risk_metrics['Volatility'] > 0.02 else 'Limited price movement'}
        * {'Potential for significant drawdowns' if risk_metrics['Trend'] == 'Downward' else 'Overvaluation risk'}

        5. Investment Implications:
        ------------------------
        {'Consider position sizing and stop-loss orders due to high volatility' if risk_metrics['Volatility'] > 0.02 else 'Suitable for longer-term position holding'}
        """
        return report
