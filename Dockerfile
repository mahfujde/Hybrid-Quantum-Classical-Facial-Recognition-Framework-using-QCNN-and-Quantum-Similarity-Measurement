# QuantumFace — CPU-only image for the API, tests, and experiments.
# RESEARCH/EDUCATION prototype; quantum work runs on local simulators.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# opencv-python-headless needs libglib at runtime; keep the layer minimal.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (avoids the large CUDA wheels pulled on x86_64).
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

COPY QuantumFace/requirements.txt /app/QuantumFace/requirements.txt
RUN pip install --no-cache-dir -r /app/QuantumFace/requirements.txt

# App code + local datasets (so the API/tests run fully offline).
COPY . /app

# Non-root user for safety.
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
# Default: serve the FastAPI app. Override for the dashboard or tests, e.g.:
#   docker run --rm quantumface python -m pytest QuantumFace/tests -q
#   docker run --rm -p 8501:8501 quantumface streamlit run QuantumFace/frontend/dashboard.py
CMD ["uvicorn", "QuantumFace.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
