FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Create sources list to enable non-free, then install dependencies
RUN echo "deb http://deb.debian.org/debian bookworm main contrib non-free" > /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y libmagic1 unrar --no-install-recommends

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
