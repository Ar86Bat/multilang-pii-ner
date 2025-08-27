FROM python:3.10-slim

WORKDIR /work

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir -r requirements.txt

RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

COPY ./api ./api

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
