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
        self.historical_data = self.load_historical_data(historical_data_path)
        self.forecast_periods = 180  # Default forecast period
        self.confidence_level = 0.95  # Default confidence level
        
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
    
    def load_historical_data(self, historical_data_path):
        """Load historical stock data."""
        try:
            data = pd.read_csv(historical_data_path)
            data['Date'] = pd.to_datetime(data['Date'])
            return data
        except Exception as e:
            print(f"Error loading historical data from {historical_data_path}: {e}")
            raise

    def generate_forecast(self):
        """Generate forecasted stock prices with confidence intervals."""
        # Prepare data for forecasting
        scaled_forecast = []
        seq_length = 90  # Model sequence length
        close_prices_scaled = self.historical_data['Close'].values[-seq_length:]
        
        # Generate forecasted values
        for _ in range(self.forecast_periods):
            input_seq_reshaped = close_prices_scaled.reshape(1, seq_length, 1)
            pred = self.model.predict(input_seq_reshaped)[0, 0]
            scaled_forecast.append(pred)
            close_prices_scaled = np.append(close_prices_scaled[1:], pred).reshape(seq_length)

        # Calculate conservative confidence intervals based on historical volatility
        forecast_std = np.std(self.historical_data['Close'].pct_change().dropna())
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)

        last_known_price = self.historical_data['Close'].iloc[-1]
        max_allowed_price = last_known_price * 1.15  # Max 15% increase
        min_allowed_price = last_known_price * 0.85  # Max 15% decrease
        forecasted_prices = np.clip(scaled_forecast, min_allowed_price, max_allowed_price)

        # Generate forecast dates and confidence intervals
        last_date = self.historical_data['Date'].max()
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=self.forecast_periods, freq='D')
        
        confidence_intervals = {
            'lower': np.maximum(forecasted_prices - (z_score * forecast_std * last_known_price), min_allowed_price),
            'upper': np.minimum(forecasted_prices + (z_score * forecast_std * last_known_price), max_allowed_price)
        }

        return pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecasted_prices,
            'Lower_CI': confidence_intervals['lower'],
            'Upper_CI': confidence_intervals['upper']
        })

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
