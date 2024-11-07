import unittest
import pandas as pd
import os
import tempfile
from datetime import datetime
import numpy as np
from scripts.data_handler import DataHandler

class TestDataHandler(unittest.TestCase):
    def setUp(self):
        # Create a temporary CSV file with test data
        self.test_data = '''Date,Open,High,Low,Close,Adj Close,Volume
2015-01-02 00:00:00+00:00,14.857999801635742,14.883333206176758,14.21733283996582,14.620667457580566,14.620667457580566,71466000
2015-01-05 00:00:00+00:00,14.303333282470703,14.433333396911621,13.810667037963867,14.005999565124512,14.005999565124512,80527500
2015-01-06 00:00:00+00:00,14.003999710083008,14.279999732971191,13.61400032043457,14.085332870483398,14.085332870483398,93928500
2015-01-07 00:00:00+00:00,14.223333358764648,14.3186674118042,13.985333442687988,14.063332557678223,14.063332557678223,44526000'''
        
        self.temp_dir = tempfile.mkdtemp()
        self.test_file_path = os.path.join(self.temp_dir, 'test_data.csv')
        
        with open(self.test_file_path, 'w') as f:
            f.write(self.test_data)
            
        self.data_handler = DataHandler(self.test_file_path)

    def tearDown(self):
        # Clean up temporary files
        os.remove(self.test_file_path)
        os.rmdir(self.temp_dir)

    def test_load_data(self):
        # Test if data is loaded correctly
        df = self.data_handler.load_data()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 4)  # Check number of rows
        self.assertEqual(len(df.columns), 6)  # Check number of columns
        self.assertTrue(isinstance(df.index, pd.DatetimeIndex))  # Check if index is DatetimeIndex

    def test_normalize_data(self):
        # Test data normalization
        df = self.data_handler.load_data()  # First load the data
        self.data_handler.df = df  # Explicitly set the df attribute
        normalized_df = self.data_handler.normalize_data()
        
        # Check if all numerical columns are normalized (mean ≈ 0, std ≈ 1)
        numerical_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in numerical_columns:
            self.assertAlmostEqual(normalized_df[col].mean(), 0, places=1)
            # Use a more flexible assertion for standard deviation
            self.assertTrue(0.8 <= normalized_df[col].std() <= 1.2, 
                          f"Standard deviation for {col} is outside acceptable range")

    def test_invalid_file_path(self):
        # Test handling of invalid file path
        invalid_handler = DataHandler("nonexistent_file.csv")
        with self.assertRaises(Exception):
            invalid_handler.load_data()

    def test_normalize_before_load(self):
        # Test normalization before loading data
        new_handler = DataHandler(self.test_file_path)
        # Ensure the data attribute is None before trying to normalize
        new_handler.data = None
        with self.assertRaises(ValueError):
            new_handler.normalize_data()

if __name__ == '__main__':
    unittest.main()