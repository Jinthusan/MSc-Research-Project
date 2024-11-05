import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from tqdm import tqdm

nltk.download('vader_lexicon')  # Ensure the VADER lexicon is downloaded

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lower case
    return text

def preprocess_data(data):
    if isinstance(data, str):  # If data is a file path
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):  # If data is already a DataFrame
        df = data
    else:
        raise ValueError("Input should be a file path or a pandas DataFrame")

    # Drop rows with missing values in 'review_content' or 'rating'
    df.dropna(subset=['review_content', 'rating'], inplace=True)

    # Convert review_rating to numeric
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)

    # Preprocess review text
    df['review_content'] = df['review_content'].apply(preprocess_text)

    # Sentiment analysis using VADER
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = []
    for text in tqdm(df['review_content']):
        sentiment_scores.append(sia.polarity_scores(text)['compound'])

    df['sentiment_score'] = sentiment_scores

    # Create a binary sales_quality target variable based on sentiment
    df['sales_quality'] = df['sentiment_score'].apply(lambda x: 1 if x >= 0 else 0)

    return df

if __name__ == "__main__":
    data_path = "../data/amazon.csv"
    processed_data = preprocess_data(data_path)
    print(processed_data.head(10))  # Display the first 10 records
