FROM python:3.11-slim

WORKDIR /app

RUN sed -i 's/main/main contrib non-free/g' /etc/apt/sources.list.d/debian.sources

RUN apt-get update && \
    apt-get install -y libmagic1 unrar --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
