from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(conf_matrix, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

def plot_accuracy(accuracy, title='Model Accuracy'):
    plt.figure(figsize=(6, 4))
    plt.bar(['Accuracy'], [accuracy], color='skyblue')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.show()

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

def train_and_evaluate_model(X, y, model_path):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a classifier
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    print("Model training completed.")
    
    # Predict sales quality
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    warnings.filterwarnings("ignore")
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division=0)
    
    print(f'Accuracy: {accuracy}')
    print(conf_matrix)
    print(class_report)
    
    # Save the model to the models directory
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}.")
    
    # Plotting the results
    plot_accuracy(accuracy)
    plot_confusion_matrix(conf_matrix)
    
    return {
        "accuracy": accuracy,
        "conf_matrix": conf_matrix,
        "class_report": class_report
    }
