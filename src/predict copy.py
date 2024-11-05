import pandas as pd

def predict_sales_quality(df, model, tfidf):
    combined_features = df['review_content'].fillna('') + ' ' + df['rating'].astype(str)
    df['predicted_sales_quality'] = model.predict(tfidf.transform(combined_features))

    for idx, row in df.head(10).iterrows():
        print(f"Product ID: {row['product_id']}")
        print(f"Review Content: {row['review_content']}")
        print(f"Review Rating: {row['rating']}")
        print(f"Sentiment Score: {row['sentiment_score']:.3f}")
        print(f"Predicted Sales Quality: {row['predicted_sales_quality']}")
        print("="*50)
