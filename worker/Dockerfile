FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Enable non-free repos and install dependencies needed by python-magic and rarfile
RUN sed -i 's/ main/ main contrib non-free/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y libmagic1 unrar --no-install-recommends

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
