from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import joblib
from app.utils import preprocess_pipeline
from app.model import extract_keywords
from app.config import MODEL_PATH, VECTORIZER_PATH, STATIC_DIR

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


class TextRequest(BaseModel):
    text: str


def create_app():
    app = FastAPI(title="Sentiment Analyzer API", version="1.0")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def read_index():
        return FileResponse(f"{STATIC_DIR}/templates/index.html")

    @app.post("/predict")
    def predict_sentiment(request: TextRequest):
        clean_text = preprocess_pipeline(request.text)
        vect_text = vectorizer.transform([clean_text])
        pred = model.predict(vect_text)[0]
        label = "Positive" if pred == 1 else "Negative"
        keywords = extract_keywords(clean_text, vectorizer, model, top_n=5)

        return JSONResponse({"sentiment": label, "keywords": keywords})

    return app
