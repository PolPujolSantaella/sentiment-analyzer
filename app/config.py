import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "../models/best_model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "../models/vectorizer.joblib")
STATIC_DIR = os.path.join(BASE_DIR, "../static")