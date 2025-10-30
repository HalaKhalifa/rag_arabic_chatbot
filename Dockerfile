FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ragchat ./ragchat
COPY data ./data
ENV PYTHONPATH=/app

CMD ["python", "-m", "ragchat.chat_cli"]
