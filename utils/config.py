import os

# Define file paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'amazon.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sentiment_model.pkl')

# You can add more configurations here if needed
