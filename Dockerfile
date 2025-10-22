FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# ✅ Installazione corretta dei pacchetti Python
RUN pip install --upgrade pip && pip install \
    numpy<2 \
    flask \
    transformers \
    sentence-transformers \
    torch \
    torchvision \
    torchaudio \
    opencv-python \
    pillow \
    psutil

# ✅ Espone la porta Flask correttamente
EXPOSE 5000

CMD ["python3", "app.py"]
