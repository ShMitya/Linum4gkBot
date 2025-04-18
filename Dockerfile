FROM python:3.11-slim

# Копируем код и модель
COPY . /app
WORKDIR /app

# Устанавливаем зависимости
RUN apt-get update && apt-get install -y wget unzip curl gnupg libglib2.0-0 libnss3 libgconf-2-4 libfontconfig1 libxss1 libappindicator3-1 libasound2 libxtst6 libatk-bridge2.0-0 libgtk-3-0 libx11-xcb1 libxcb-dri3-0 libgbm1 xdg-utils --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Установка Google Chrome
RUN apt-get update && apt-get install -y wget gnupg unzip curl && \
    wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && apt-get install -y google-chrome-stable

# Устанавливаем совместимую версию ChromeDriver для Chrome 135
RUN curl -sSL "https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/135.0.7049.95/linux64/chromedriver-linux64.zip" \
    -o /tmp/chromedriver.zip && \
    unzip /tmp/chromedriver.zip -d /tmp/ && \
    mv /tmp/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver && \
    chmod +x /usr/local/bin/chromedriver && \
    rm -rf /tmp/chromedriver*

# Установка ChromeDriver версии 114.0.5735.90 (пример)
#RUN DRIVER_VERSION="114.0.5735.90" && \
#    DRIVER_URL="https://chromedriver.storage.googleapis.com/${DRIVER_VERSION}/chromedriver_linux64.zip" && \
#    curl -sSL "$DRIVER_URL" -o /tmp/chromedriver.zip && \
#    unzip /tmp/chromedriver.zip -d /usr/local/bin/ && \
#    chmod +x /usr/local/bin/chromedriver && \
#    rm /tmp/chromedriver.zip

# Папка temp
RUN mkdir -p /tmp && chmod -R 1777 /tmp
ENV TMPDIR=/tmp

# Установка Python-зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Переменные окружения
ENV BOT_TOKEN=""
ENV EMAIL=""
ENV PASSWORD=""
ENV ALLOWED_USERS=""

# Запуск бота
CMD ["python", "Linum4gkBot.py"]
