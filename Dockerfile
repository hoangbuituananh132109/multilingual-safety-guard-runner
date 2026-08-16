ARG PYTORCH_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/project/cache/huggingface \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/safety-guard
COPY requirements.txt ./
RUN python -m pip install --upgrade pip wheel setuptools && \
    python -m pip install -r requirements.txt

COPY . /opt/safety-guard
WORKDIR /workspace/project
ENTRYPOINT ["/usr/bin/tini", "--", "python", "/opt/safety-guard/run.py"]
CMD ["--help"]
