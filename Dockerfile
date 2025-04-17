FROM python:3.11-slim

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    wget curl unzip gnupg ca-certificates \
    fonts-liberation libnss3 libatk-bridge2.0-0 libxss1 libasound2 libx11-xcb1 libxcomposite1 libxcursor1 libxdamage1 libxi6 libxtst6 libxrandr2 libgbm1 \
    chromium chromium-driver

# Установка python-библиотек
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и модель
COPY . /app
WORKDIR /app

# Переменные окружения
ENV BOT_TOKEN=""
ENV EMAIL=""
ENV PASSWORD=""
ENV ALLOWED_USERS=""

# Запуск бота
CMD ["python", "Linum4gkBot.py.py"]
