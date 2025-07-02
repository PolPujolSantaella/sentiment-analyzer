import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Tuple, List, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def vectorize_text(train_texts: List[str], test_texts: List[str], 
                   max_features: int = 7000, ngram_range: Tuple[int, int] = (1, 3),
                   stop_words: str = 'english',
                   min_df: int = 3,
                   max_df: float = 0.85
                  ) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    Vectorizes the text data using TF-IDF vectorization.

    Args:
        train_texts (List[str]): _description_
        test_texts (List[str]): _description_
        max_features (int, optional): _description_. Defaults to 5000.
        ngram_range (Tuple[int, int], optional): _description_. Defaults to (1, 2).

    Returns:
        X_train, X_test, fitted vectorizer
    """
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
        min_df=min_df,
        max_df=max_df
    )
    
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer


def train_model(X_train, y_train, model_type='logistic', cv_folds=5, random_state=42
                   ) -> Union[LogisticRegression, MultinomialNB, LinearSVC]:
    """
    Perform grid search with cross-validation to find the best hyperparameters.
    
    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        model_type (str, optional): Model type ('logistic', 'naive_bayes', 'svm').
        cv_folds (int, optional): Number of cross-validation folds.
        random_state (int, optional): Random seed.
    
    Returns:
        Best trained model and best hyperparameters found.
    """
    
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=500, random_state=random_state)

        param_grid = [
            {
                'C': [0.01, 0.1, 1],                # Regularization strength
                'penalty': ['l1'],                  # Tipo de regularización
                'solver': ['saga'],            # Solver compatible con l1
            },
            {
                'C': [0.01, 0.1, 1],                # Regularization strength
                'penalty': ['l2'],                  # Tipo de regularización
                'solver': ['lbfgs', 'saga']  # Solvers compatibles con l2
            }
     ]
    elif model_type == 'naive_bayes':
        model = MultinomialNB()
        param_grid = {
            'alpha': [0.01, 0.1, 0.5, 1, 2],     # Suavizado
            'fit_prior': [True, False]                      # ¿Ajustar priors?
        }
    elif model_type == 'svm':
        model = LinearSVC(random_state=random_state, max_iter=5000)
        param_grid = [
        {  
            'C': [0.01, 0.1, 1, 10],
            'loss': ['hinge'],
            'dual': [True]
        },
        {   
            'C': [0.01, 0.1, 1, 10],
            'loss': ['squared_hinge'],
            'dual': [True, False]
        }
    ]
    else:
        raise ValueError("Unsupported model type. Choose from 'logistic', 'naive_bayes', or 'svm'.")

    grid = GridSearchCV(model, param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"Best parameters for {model_type} model: {grid.best_params_}")
    print(f"Best cross-validation score: {grid.best_score_:.4f}")
    
    return grid.best_estimator_, grid.best_params_

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, labels: List[str] = ["Neg", "Pos"]) -> float:
    """
    Evaluate the model on the test set and print the accuracy, classification report, and confusion matrix.

    Args:
        model (_type_): _description_
        X_test (np.ndarray): The test data.
        y_test (np.ndarray): The test labels.
        labels (List[str], optional): The labels for the confusion matrix. Defaults to ["Neg", "Pos"].
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
    
    return acc
    
def extract_keywords(text: str, vectorizer: TfidfVectorizer, model, top_n: int = 5) -> List[str]:
    """
    Extract the top N keywords from the text based on the model's coefficients.

    Args:
        text (str): The input text.
        vectorizer (TfidfVectorizer): The TF-IDF vectorizer.
        model (Union[LogisticRegression, MultinomialNB, LinearSVC]): The trained model.
        top_n (int, optional): Number of top keywords to extract. Defaults to 5.
    """
    
    text_vectorized = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    weights = text_vectorized.toarray().flatten()
    
    present_indices = np.where(weights > 0)[0]
    
    if len(present_indices) == 0:
        return []

    coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
    
    importance = weights[present_indices] * coef[present_indices]
    top_indices_local = np.argsort(importance)[-top_n:][::-1]
    top_indices_global = present_indices[top_indices_local]
    
    keywords = feature_names[top_indices_global]
    
    return keywords.tolist()
    
    
    