FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models
WORKDIR /work
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir -r requirements.txt
# Optional: warm up HF cache so the first notebook cell is faster (tiny no-op call)
RUN python3 - <<'PY'
from datasets import load_dataset; print("Datasets OK")
from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("xlm-roberta-base")
print("HF cache primed")
PY
EXPOSE 8888
CMD ["python3","-m","jupyter","lab","--ip=0.0.0.0","--port=8888","--no-browser","--NotebookApp.token=","--NotebookApp.password="]
