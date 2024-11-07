import os
import tempfile
import pandas as pd
import numpy as np
import unittest
from scripts.financial_eda import FinancialEDA

class TestFinancialEDA(unittest.TestCase):

    def setUp(self):
        # Create a temporary CSV file with sample data
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.csv')
        
        # Create sample data
        date_range = pd.date_range(start="2020-01-01", periods=750, freq='D')
        
        close_prices = [100]  # First day price
        close_prices.append(105)  # Second day price (5% increase)
        
        # Generate remaining prices
        for x in range(2, 750):
            close_prices.append(100 + (x * 0.1) + (x % 20))
        
        data = pd.DataFrame({
            'Date': date_range,
            'Close': close_prices
        })
        
        # Save to CSV
        data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()

        # Initialize the FinancialEDA class
        self.eda = FinancialEDA(self.temp_file.name)
        self.eda.df['Daily Change'] = self.eda.df['Close'].pct_change()

    def test_daily_percentage_change(self):
        daily_changes = self.eda.df['Daily Change']
        self.assertAlmostEqual(daily_changes.iloc[1], 0.05, places=2)

    def test_rolling_stats(self):
        self.eda.rolling_stats(window=2)

    def tearDown(self):
        try:
            os.remove(self.temp_file.name)
        except PermissionError:
            self.temp_file.close()
            os.remove(self.temp_file.name)

if __name__ == '__main__':
    unittest.main()