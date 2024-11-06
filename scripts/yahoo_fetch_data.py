import yfinance as yf

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