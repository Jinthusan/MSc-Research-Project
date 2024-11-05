import unittest
import pandas as pd
from src.preprocess import preprocess_data

class TestPreprocessData(unittest.TestCase):
    def test_preprocess_data(self):
        # Create a small sample DataFrame for testing
        data = {
            'review_content': ['Good product', 'Bad product', 'Average quality', 'Excellent item', 'Poor performance'],
            'rating': [5, 1, 3, 5, 1]
        }
        df = pd.DataFrame(data)
        
        # Call preprocess_data with DataFrame directly
        processed_data = preprocess_data(df)
        
        self.assertIn('sales_quality', processed_data.columns)
        self.assertFalse(processed_data.isnull().values.any())
        print(processed_data.head(10))  # Display the first 10 records for debugging

if __name__ == "__main__":
    unittest.main()
