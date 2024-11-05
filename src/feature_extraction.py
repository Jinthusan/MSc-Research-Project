from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(df):
    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=5000)

    # Combine text features and numeric features
    combined_features = df['review_content'].fillna('') + ' ' + df['rating'].astype(str)

    # Fit and transform TF-IDF on combined features
    X = tfidf.fit_transform(combined_features)
    y = df['sales_quality']
    
    print("Feature extraction completed.")
    
    return X, y, tfidf
