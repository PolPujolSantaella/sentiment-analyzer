FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc python3-dev libffi-dev libssl-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requeriments-api.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requeriments-api.txt

COPY ./app ./app
COPY ./models ./models
COPY ./static ./static

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
