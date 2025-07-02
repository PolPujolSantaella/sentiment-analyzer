# Documentation 

## 📄 Text Preprocessing Documentation

This document explains the preprocessing pipeline used in the Sentiment Analyzer project. Preprocessing is a critical step in Natural Language Processing (NLP) that transforms raw text into a clean, normalized and structurd format suitable for Machine Learning models.

## 🛠️ Preprocessing Steps Overview

The pipeline in `app/utils.py` includes the following techniques:

- ✅ Text Cleaning
- ✅ Tokenization (NLTK)
- ✅ Stopwords Removal (NLTK)
- ✅ Lemmatization (spaCy)

This steps reduce noise, normalize language, and improve model performance.

### Clean Text

Remove unnecessary elements from raw text to standarize it:

- Remove HTML tags
- Remove punctuation, numbers, special characters (keep only letters)
- Convert to lowercase
- Remove extra whitespaces

### Tokenize

Split text into individual words or tokens using NLTK's tokenizer.

Why? Machine Learning models work with structured data; tokenization breaks sentences into unirs that algorithms can understand.

Example:

```bash
tokenize("This is a project.")
# Output: ['This', 'is', 'a', 'project', '.']
```

### Remove Stopwords

Eliminate common words (stopwords) that carry little meaningful information for sentiment analysis, for example:
`["the", "is", "in", "at", "which", "on", "and"]`

This step reduce dimensionality, focus on meaningful words and improve model signal-to-noise ratio.

### Lemmatize

Recuce words to their root form (lemma) using SpaCy.

- Groups similar words (e.g., "running", "ran" --> "run")
- Reduces vocabulary size
- Preserves semantic meaning

SpaCy model used: en_core_web_sm (downloaded if not available)
Pronouns (-PRON-) are excluded


## 📄 Machine Learning & Text Vectorization Module Documentation

This part describes the core machine learning and festure extraction techniques implemented in the `app/model.py` file of the project.

The focus in on preparing text data, building classification models, evaluating performance, and extracting interpretable keywords from model predictions.

### 🧩 Module Overview

- ✅ TF-IDF Text Vectorization
- ✅ Model Training (Logistic Regression, Naive Bayes, SVM)
- ✅ Model Evaluation with Visuals
- ✅ Keyword Extraction Based on Model Coefficients

### Vectorize Text

Converts raw text data into numerical feature vectors using TF-IDF (Term Frequency - Inverse Document Frequency).

TF-IDF Captures word importances based on frequency and uniqueness, uses both unigrams (1 word) and bigrams (2 consecutive words).

Returns vectorized features and a fitted `TfidVectorizer` for futer transformations

### Train Model

Trains a classification model based on specified type.

Supported Models:

- LR (Logistic Regression)
- Naive Bayes
- SVM (Linear Support Vector Machine)

Returns a trained classifier ready for predictions

### Evaluate Model

Evaluates model performance on test data and provides:

- Accuracy Score
- Classification report (precision, recall, F1)
- Confusion Matrix heatmap visualization
