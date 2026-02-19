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
    pip install streamlit requests pandas numpy

EXPOSE 8501

CMD ["sh", "-c", "cd frontend && streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"]
