FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ .

CMD ["python", "main.py"]
# docker buildx build --platform linux/amd64 -f etc/docker/Dockerfile -t miguelgrigorio27/is-tiffany-detection:latest . --push
