FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY ./ /app/

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install fastapi uvicorn pydantic scikit-learn pandas numpy joblib python-multipart requests

EXPOSE 8000

CMD ["sh", "-c", "cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000"]
