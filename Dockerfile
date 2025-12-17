FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY backend_unet_resnet.py .
COPY models_unet_resnet.py .
COPY preprocessing.py .

COPY weights/ weights/
EXPOSE 8000
CMD uvicorn backend_unet_resnet:app --host 0.0.0.0 --port ${PORT:-8000}
