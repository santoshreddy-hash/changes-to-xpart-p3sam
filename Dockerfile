# partseg runtime image. NOT built or tested on the development machine (Windows, no Docker):
# treat it as a starting point and verify with:  docker build -t partseg . && docker run --gpus all partseg doctor
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev build-essential git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3.10 /usr/local/bin/python

WORKDIR /app
COPY requirements-partseg.txt .
# CUDA-specific wheels first (match the CUDA version of the base image)
RUN python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118 \
    && python -m pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html \
    && python -m pip install -r requirements-partseg.txt

COPY . .
# model weights are downloaded on first use; mount a volume to keep them between runs
ENV HF_HOME=/models/hf
VOLUME ["/models", "/data"]
ENTRYPOINT ["python", "-m", "partseg"]
CMD ["--help"]
