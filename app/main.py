import os 
import sys
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import joblib

from app.utils import preprocess_pipeline

sys.path.append(os.path.abspath('..'))

app = FastAPI(title="API + Frontend Sentiment Analyzer", version="1.0")

model = joblib.load('models/best_model.joblib')
vectorizer = joblib.load('models/vectorizer.joblib')

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

class TextRequest(BaseModel):
    text: str
    
@app.post("/predict")
def predict_sentiment(request: TextRequest):
    clean_text = preprocess_pipeline(request.text)
    vect_text = vectorizer.transform([clean_text])
    pred = model.predict(vect_text)[0]
    label = "Positive" if pred == 1 else "Negative"
    return JSONResponse({"sentiment": label})