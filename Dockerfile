# ⚙️ Base image con Python e supporto per PyTorch + CUDA
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 📁 Imposta la directory di lavoro
WORKDIR /app

# 🧩 Installa dipendenze di sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 📜 Copia i file del progetto
COPY . /app

# 📦 Installa le dipendenze Python
RUN pip install --upgrade pip && pip install \
    flask \
    transformers \
    sentence-transformers \
    torch \
    torchvision \
    torchaudio \
    opencv-python \
    pillow \
    psutil \
    numpy<2

# 🚪 Espone la porta Flask
EXPOSE 5000

# ▶️ Comando di avvio
CMD ["python3", "app.py"]

