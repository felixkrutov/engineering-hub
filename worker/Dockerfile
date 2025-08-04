FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install system dependencies needed by python-magic
RUN apt-get update && apt-get install -y libmagic1

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
