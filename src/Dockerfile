FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
WORKDIR /app
RUN apt-get update && apt-get install -y python3 python3-pip git
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY serve/ ./serve/
CMD ["python3", "serve/app.py", "--base_model", "meta-llama/Llama-2-7b", "--adapter_path", "outputs/lora-adapter"]