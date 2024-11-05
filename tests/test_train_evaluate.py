import unittest
import pandas as pd
from src.train_evaluate import train_and_evaluate_model

class TestTrainEvaluateModel(unittest.TestCase):
    def test_train_and_evaluate_model(self):
        # Prepare a small sample DataFrame for testing
        data = {
            'review_content': ['Good', 'Bad', 'Average', 'Excellent', 'Poor'],
            'sales_quality': [1, 0, 0, 1, 0]
        }
        df = pd.DataFrame(data)
        
        # Simulate the preprocessing step
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['review_content'])
        
        X = tfidf_matrix.toarray()
        y = df['sales_quality']
        
        # Save model path
        model_path = "models/test_sales_quality_model.pkl"
        
        results = train_and_evaluate_model(X, y, model_path)
        
        self.assertIn("accuracy", results)
        self.assertIn("conf_matrix", results)
        self.assertIn("class_report", results)

if __name__ == "__main__":
    unittest.main()
