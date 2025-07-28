# Используем официальный Python-образ
FROM python:3.11-slim

# Устанавливаем ffmpeg и необходимые системные зависимости
RUN apt-get update && \
    apt-get install -y ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы приложения
COPY requirements.txt ./
COPY speech_to_sentiment_app.py ./
COPY temp_audio ./temp_audio

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт для Streamlit
EXPOSE 8501

# Запускаем Streamlit-приложение
CMD ["streamlit", "run", "speech_to_sentiment_app.py", "--server.port=8501", "--server.address=0.0.0.0"] 