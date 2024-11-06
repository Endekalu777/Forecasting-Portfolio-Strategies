import yfinance as yf
import logging
import os
from datetime import datetime

# Create log folder if it doesnot exist
if not os.path.exists("../logs"):
    os.makedirs("../logs")

# Configure logging
log_filename = os.path.join('logs', f'stock_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename)
    ]
)

# Download historical data for TSLA, BND, and SPY
tickers = ['TSLA', 'BND', 'SPY']
data = yf.download(tickers, start="2015-01-01", end="2023-01-01", group_by='ticker')

# Save each ticker's data to separate CSV files
for ticker in tickers:
    # Extract data for this ticker
    ticker_data = data[ticker]
    
    # Save to CSV
    filename = f'{ticker}_historical_data.csv'
    ticker_data.to_csv(filename)
    print(f'Saved {filename}')