# syntax=docker/dockerfile:1
# MVP Trợ lý Thị giác AI — image inference cho NVIDIA T4 (sm_75).
#
# Nguyên tắc (README "Việc còn lại"): CODE vào image, WEIGHTS/DỮ LIỆU MOUNT NGOÀI:
#   models/hf  (~19 GB)  -> /home/ubuntu/ai-assistant/models/hf   (HF_HOME)
#   .env       (API key) -> /home/ubuntu/ai-assistant/.env          :ro
#   certs/     (TLS)     -> /home/ubuntu/ai-assistant/certs         (tự sinh nếu thiếu)
#   logs/                -> /home/ubuntu/ai-assistant/logs
# Đường dẫn trong container GIỐNG HOST vì api_main.py ghim ROOT tuyệt đối.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/ubuntu/ai-assistant/models/hf \
    TOKENIZERS_PARALLELISM=false

# gcc + libc6-dev: triton JIT lúc runtime (SDPA/inductor) — với
# --no-install-recommends phải khai báo rõ libc6-dev, thiếu là mất stdlib.h
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libsndfile1 libgomp1 openssl gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /home/ubuntu/ai-assistant

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY api_main.py README.md ./
COPY scripts/ scripts/
COPY web/ web/
COPY assets/ assets/
COPY benchmarks/ benchmarks/
COPY tests/ tests/
COPY deploy/ deploy/

RUN chmod +x scripts/run_demo_server.sh

EXPOSE 8443 8000
ENTRYPOINT ["./scripts/run_demo_server.sh"]
