from fastapi import FastAPI
from pydantic import BaseModel
import joblib

import os 
import sys

sys.path.append(os.path.abspath('..'))

app = FastAPI(title="API Sentiment Analysis", version="1.0")

model = joblib.load('models/best_model.joblib')
vectorizer = joblib.load('models/vectorizer.joblib')

class TextRequest(BaseModel):
    text: str
    
@app.get("/")
def root():
    return {"message": "API for sentiment analysis. Use /predict to analyze text sentiment."}

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    clean_text = request.text.lower()  # Simple text cleaning
    vect_text = vectorizer.transform([clean_text])
    pred = model.predict(vect_text)[0]
    label = "Positive" if pred == 1 else "Negative"
    return {"sentiment": label}