FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Large wheels over a slow link can stall past pip's 15s default and
    # abort the whole build with a ReadTimeoutError. Be patient instead.
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch BEFORE requirements.txt.
#
# sentence-transformers depends on torch, and torch's default PyPI wheel
# for Linux bundles the entire CUDA stack — a 427MB wheel plus several GB
# of nvidia-cublas / cudnn / nccl / triton packages. DriftScope only ever
# runs the ~80MB all-MiniLM-L6-v2 model on CPU, so none of that is used;
# it just made the image enormous and the download prone to timing out.
#
# The +cpu local version exists only on PyTorch's own index, so pinning it
# exactly guarantees we get the CPU build even with PyPI as a fallback for
# torch's own dependencies. Installing it first means the later
# `-r requirements.txt` sees torch as already satisfied.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      torch==2.9.1+cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the image so the first real
# request doesn't pay an ~80MB Hugging Face download + load penalty.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
