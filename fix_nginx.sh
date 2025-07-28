#!/bin/bash

echo "=== ЭКСТРЕННОЕ ИСПРАВЛЕНИЕ NGINX ==="

# 1. Полная остановка всех процессов nginx
echo "Остановка всех процессов nginx..."
sudo pkill -f nginx
sudo systemctl stop nginx

# 2. Создание минимальной конфигурации
echo "Создание чистой конфигурации..."
sudo mkdir -p /etc/nginx/conf.d

# Бэкап основного файла на всякий случай
sudo cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup

# Создание чистого nginx.conf, который включает только одну папку
sudo tee /etc/nginx/nginx.conf > /dev/null << 'EOF'
user www-data;
worker_processes auto;
pid /run/nginx.pid;
include /etc/nginx/modules-enabled/*.conf;

events {
    worker_connections 768;
}

http {
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    ssl_protocols TLSv1 TLSv1.1 TLSv1.2 TLSv1.3; # Dropping SSLv3, ref: POODLE
    ssl_prefer_server_ciphers on;

    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;

    gzip on;

    # Включаем только наш конфиг из ОДНОЙ папки
    include /etc/nginx/conf.d/*.conf;
}
EOF

# 3. Создание нашей точной конфигурации сайта
sudo tee /etc/nginx/conf.d/default.conf > /dev/null << 'EOF'
server {
    listen 80 default_server;
    server_name 51.158.76.113;

    # Правило для Проекта 1
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Правило для Проекта 2
    location /hub/ {
        proxy_pass http://127.0.0.1:8081/; 
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        rewrite ^/hub/(.*)$ /$1 break;
    }
}
EOF

# 4. Очистка всех других возможных конфигураций
echo "Очистка старых папок..."
sudo rm -f /etc/nginx/sites-enabled/*
sudo rm -f /etc/nginx/sites-available/*

# 5. Проверка конфигурации
echo "Проверка конфигурации..."
sudo nginx -t

if [ $? -eq 0 ]; then
    echo "Конфигурация корректна. Запуск nginx..."
    sudo systemctl start nginx
    sudo systemctl enable nginx
    echo "Nginx запущен!"
else
    echo "ОШИБКА в конфигурации nginx! Восстанавливаем из бэкапа..."
    sudo cp /etc/nginx/nginx.conf.backup /etc/nginx/nginx.conf
    exit 1
fi

echo "ВСЕ ГОТОВО!"
