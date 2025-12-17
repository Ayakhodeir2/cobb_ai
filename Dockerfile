FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY backend_clinical.py .
COPY models_unet_resnet.py .
COPY preprocessing.py .

# Copy model weights directory
COPY weights/ weights/

# Expose port
EXPOSE 8000

# Run the application (Railway will provide PORT env var)
CMD uvicorn backend_clinical:app --host 0.0.0.0 --port ${PORT:-8000}
