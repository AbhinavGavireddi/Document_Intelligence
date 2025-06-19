# Base image
FROM python:3.10-slim

RUN useradd -m -u 1000 user
USER user

# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    # for hnswlib (needed for OpenMP)
    libgomp1 \
    curl \
    git && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install uv && \
    uv pip install --no-cache-dir -r requirements.txt

# Download models (if needed at build time)
RUN curl -L https://github.com/opendatalab/MinerU/raw/dev/scripts/download_models_hf.py -o download_models_hf.py && \
    python download_models_hf.py

# Copy application code
COPY src/ ./src/

# COPY tests/ ./tests/
COPY app.py .

# Expose Streamlit port
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# Start Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
