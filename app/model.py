import pandas as pd 
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def vectorize_text(train_texts, test_texts, max_festures=5000):
    """
    Vectorizes the text data using TF-IDF vectorization.

    Args:
        train_texts (_type_): _description_
        test_texts (_type_): _description_
        max_festures (int, optional): _description_. Defaults to 5000.
    """
    
    vectorizer = TfidfVectorizer(max_features=max_festures, ngram_range=(1,2))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer

def train_model(X_train, y_train, model_type='logistic'):
    """
    Train the model based on the specified type.

    Args:
        X_train (_type_): _description_
        y_train (_type_): _description_
        model_type (str, optional): _description_. Defaults to 'logistic'.
    """
    
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'naive_bayes':
        model = MultinomialNB()
    elif model_type == 'svm':
        model = LinearSVC()
    else:
        raise ValueError("Unsupported model type. Choose from 'logistic', 'naive_bayes', or 'svm'.")

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and print the accuracy, classification report, and confusion matrix.

    Args:
        model (_type_): _description_
        X_test (_type_): _description_
        y_test (_type_): _description_
    """
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    
    
    