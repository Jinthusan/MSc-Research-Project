import pandas as pd
from preprocess import preprocess_data
from feature_extraction import extract_features
from train_evaluate import train_and_evaluate_model
from predict import predict_sales_quality
import joblib

def main():
    # Load dataset
    data_path = '/Users/jinthusan/Documents/University of Derby/Research/Product Sales Prediction/data/amazon.csv'
    df = pd.read_csv(data_path)

    # Preprocess data
    df = preprocess_data(df)
    
    # Feature extraction
    X, y, tfidf = extract_features(df)

    # Train and evaluate model
    model_path = '/Users/jinthusan/Documents/University of Derby/Research/Product Sales Prediction/models/sales_quality_model.pkl'
    results = train_and_evaluate_model(X, y, model_path)
    
    # Load the trained model
    model = joblib.load(model_path)

    # Predict and display results
    predict_sales_quality(df, model, tfidf)

    print(results)  # Display model evaluation results

if __name__ == "__main__":
    main()
