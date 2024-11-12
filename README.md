# Project Structure
The project structure is organized as follows:

```
├── notebooks                        # Jupyter Notebooks for Exploratory Data Analysis (EDA) and model training
│   ├── BND_financial_eda.ipynb          # EDA for Vanguard Total Bond Market ETF (BND)
│   ├── BND_forecast.ipynb               # Forecasting BND prices
│   ├── BND_model_training.ipynb         # Model training for BND time series forecasting
│   ├── SPY_financial_eda.ipynb          # EDA for S&P 500 ETF (SPY)
│   ├── SPY_forecasting.ipynb            # Forecasting SPY prices
│   ├── SPY_model_training.ipynb         # Model training for SPY time series forecasting
│   ├── TSLA_financial_eda.ipynb         # EDA for Tesla stock (TSLA)
│   ├── TSLA_forecast.ipynb              # Forecasting TSLA prices
│   └── TSLA_model_training.ipynb        # Model training for TSLA time series forecasting
│
├── scripts                         # Python scripts for various data handling and analysis functions
│   ├── data_handler.py                 # Data loading and preprocessing functions
│   ├── financial_eda.py                # Functions for Exploratory Data Analysis (EDA)
│   ├── forecast.py                     # Functions for time series forecasting
│   ├── portfolio_optimizer.py          # Functions for optimizing portfolio based on forecasts
│   ├── time_series_model_training.py   # Model training pipeline for time series forecasting
│   └── yahoo_fetch_data.py             # Script to fetch financial data using YFinance
│
└── tests                           # Unit tests for ensuring code quality
    ├── test_data_handler.py            # Tests for data handling functions
    └── test_financial_eda.py           # Tests for EDA functions
```

# Branches
The project includes the following branches:

- **main**: The main branch with all merged changes from feature branches.
- **task1**: Branch for data preprocessing and exploratory data analysis.
- **task2**: Branch for developing time series forecasting models.
- **task3**: Branch for forecasting future market trends.
- **task4**: Branch for optimizing the investment portfolio based on forecasted data.

# Project Tasks

## Task 1: Preprocess and Explore the Data
- Load and clean historical financial data for Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY) using YFinance.
- Conduct data cleaning:
  - Check statistics to understand the data distribution.
  - Handle missing values and ensure data types are appropriate.
- Conduct Exploratory Data Analysis (EDA):
  - Visualize historical trends and volatility.
  - Analyze rolling means and volatility.
  - Detect and analyze outliers.

## Task 2: Develop Time Series Forecasting Models
- Build and evaluate time series forecasting models for TSLA.
- Model selection options:
  - **ARIMA** for univariate series without seasonality.
  - **SARIMA** for series with seasonality.
  - **LSTM** for long-term dependencies in time series data.
- Evaluate model performance using metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

## Task 3: Forecast Future Market Trends
- Use the trained models to forecast future prices for TSLA, SPY, and BND over a 6-12 month period.
- Forecast analysis:
  - Visualize forecasts alongside historical data.
  - Include confidence intervals to illustrate forecast uncertainty.
  - Provide trend and risk analysis based on the forecasted data.

## Task 4: Optimize Portfolio Based on Forecast
- Combine forecasted data for TSLA, SPY, and BND to construct a portfolio.
- Portfolio optimization:
  - Calculate returns, risks, and risk-adjusted performance metrics (e.g., Sharpe Ratio).
  - Adjust asset weights to optimize portfolio performance based on forecasted trends.
- Analyze expected return, volatility, and Sharpe Ratio, providing visualizations for cumulative return and risk-return analysis.

# Getting Started
To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Endekalu777/Forecasting-Portfolio-Strategies
   cd Forecasting-Portfolio-Strategies
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Installing packages**:
Install the necessary packages with:

    ```
    pip install -r requirements.txt
    ```

4. **Run the Notebooks**: Navigate to the notebooks folder to explore data and run analyses.

5. **Run Scripts**: Each script in the scripts folder is designed for specific tasks, from data fetching to portfolio optimization.

6. **Testing**: Run tests located in the tests folder to verify the functionality of various components.

# Contributions
**Contributions** are welcome! If you find any issues or have suggestions for improvement, feel free to create a pull request or open an issue.

# License
This project is licensed under the MIT License. See the LICENSE file for more details.

# Contact
For any questions or additional information, please contact [Endekalu.simon.haile@gmail.com](mailto:Endekalu.simon.haile@gmail.com)