# FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
# WORKDIR /app
# RUN apt-get update && apt-get install -y python3 python3-pip git
# COPY requirements.txt .
# RUN pip install -r requirements.txt
# COPY serve/ ./serve/
# CMD ["python3", "serve/app.py", "--base_model", "meta-llama/Llama-2-7b", "--adapter_path", "outputs/lora-adapter"]

FROM python:3.10-slim

# Force linux/amd64 for Apple Silicon if needed
# (buildx will handle the emulation when you build/run with --platform)
# Example: docker buildx build --platform linux/amd64 -t text2sql-service .

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run service
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
