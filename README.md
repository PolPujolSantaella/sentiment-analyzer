# 🧠 Sentiment Analyzer - Text Classification with FastAPI, ML & Docker

## Overview

Sentiment Analyzer is a production-ready, containerized web application that classifies text reviews as **Positive** or **Negatives** using Machine Learning. Built with a clean architecture, it combines:

✅ Natural Language Processing (NLP)
✅ Machine Learning (TF-IDF + Logistic Regression)  
✅ FastAPI for API backend 
✅ Docker for easy deployment 

This project demonstrates real-world skills in NLP, model deployment, and building interactive applications.

---

## 🔧 Features

- Text sentiment classification (Positive / Negative)
- Keyword extraction showing influential words
- Preprocessing pipeline with cleaning, tokenization, stopwords removal, lemmatization.
- User-friendly web interface
- Loading spinner and visual feedback
- Fully containerized with Docker
- Ready for development and production branches

---

## 📁 Project Structure

```bash
├── app/ # Python backend
│ ├── init.py
│ ├── api.py # FastAPI app
│ ├── utils.py # Preprocessing utilities
│ ├── model.py # Keyword extraction, model helpers
│
├── static/ # Frontend assets
│ ├── templates/
│ │ └── index.html
│ ├── css/
│ │ └── styles.css
│ ├── js/
│ └── app.js
│
├── models/ # Trained ML models
│ ├── best_model.joblib
│ └── vectorizer.joblib
│
├── notebooks/ # Data exploration & experiments
├── data/ # Datasets (raw and cleaned)
├── Dockerfile
├── requirements.txt
├── main.py  # Create app
└── README.md
```

---

## 🚀 Getting Started

### Requirements

- Docker installed (recommended)
or 
- Python 3.10+, `pip` package manager

---

### Run with Docker (recommended)
```bash
docker build -t sentiment-analyzer .
docker run -p 8000:8000 sentiment-analyzer
```
Access the app at: http://localhost:8000

### Run Locally (Dev Mode)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## 🧪 Model Details

- Dataset: IMDB Movie Reviews (Balanced: 25.000 Positive, 25.000 Negatives)
- Preprocessing:
    - Text cleaning (punctuation, symbols)
    - Tokenization
    - Stopword removal(NLTK)
    - Lemmatization (spaCy)
- Vectorization: TF-IDF
- Classifier: Logistic Regression
- Evaluation: Accuracy, Confusion Matrix, Precision, Recall, F1-Score

## 📄 Future Improvements

- Switch between multiple models (Naive Bayes, SVM)
- Advanced visualization for model explanations (SHAP)
- User input validation and multilingual support

## 📢 License
MIT License - Free to use and modify.

## 💡 Author
Pol - Computer Engineer passionate about AI, F1, and building impactful applications.
