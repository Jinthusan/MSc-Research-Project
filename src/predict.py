import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sentiment_distribution(df, title='Sentiment Score Distribution'):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sentiment_score'], kde=True, color='green')
    plt.title(title)
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()

def plot_sales_quality_distribution(df, title='Predicted Sales Quality Distribution'):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='predicted_sales_quality', data=df, palette='viridis')
    plt.title(title)
    plt.xlabel('Predicted Sales Quality')
    plt.ylabel('Count')
    plt.show()

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
    
    # Plot sentiment score distribution
    plot_sentiment_distribution(df)
    # Plot predicted sales quality distribution
    plot_sales_quality_distribution(df)
