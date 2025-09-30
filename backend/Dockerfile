FROM python:3.11-slim

WORKDIR /app

RUN sed -i 's/main/main contrib non-free/g' /etc/apt/sources.list.d/debian.sources

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bash \
    netcat-openbsd \
    libmagic1 \
    unrar \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x ./wait-for-it.sh

CMD ["bash", "./wait-for-it.sh", "db:5432", "--", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
