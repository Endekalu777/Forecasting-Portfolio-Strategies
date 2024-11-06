import yfinance as yf
import logging
import os
from datetime import datetime

# Create log folder if it doesnot exist
if not os.path.exists("../logs"):
    os.makedirs("../logs")

# Configure logging
log_filename = os.path.join('../logs', f'stock_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename)
    ]
)

logger = logging.getLogger(__name__)

try:
    logger.info("Starting data download process")
    
    # Download historical data for TSLA, BND, and SPY
    tickers = ['TSLA', 'BND', 'SPY']
    logger.info(f"Downloading data for tickers: {', '.join(tickers)}")
    
    data = yf.download(tickers, start="2015-01-01", end="2023-01-01", group_by='ticker')
    logger.info("Data download completed successfully")

    # Save each ticker's data to separate CSV files
    for ticker in tickers:
        try:
            # Extract data for this ticker
            ticker_data = data[ticker]

            # Create a data folder if it doesnot exist
            if not os.path.exists("../data"):
                os.makedirs("../data")

            # Save to CSV
            filename = f'../data/{ticker}_historical_data.csv'
            ticker_data.to_csv(filename)
            logger.info(f'Successfully saved {filename}')
            
        except Exception as e:
            logger.error(f'Error saving data for {ticker}: {str(e)}')

except Exception as e:
    logger.error(f"An error occurred during execution: {str(e)}")
    raise

finally:
    logger.info("Process completed")