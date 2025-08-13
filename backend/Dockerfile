FROM python:3.11-slim

WORKDIR /app

# ИСПРАВЛЕНИЕ: Эта команда корректно добавляет репозитории 'contrib' и 'non-free'
# к существующим настройкам, вместо того чтобы ломать их.
RUN sed -i 's/main/main contrib non-free/g' /etc/apt/sources.list.d/debian.sources

# Устанавливаем зависимости и сразу чистим кэш apt
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

# Запускаем скрипт через BASH
CMD ["bash", "./wait-for-it.sh", "db:5432", "--", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
