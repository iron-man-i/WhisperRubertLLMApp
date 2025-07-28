import streamlit as st
import numpy as np
import ffmpeg
from pywhispercpp.model import Model
from transformers import pipeline, AutoTokenizer
from pathlib import Path
import sys
import subprocess
import os
import requests

# Получение API-ключа Deepseek из переменных окружения
def get_deepseek_api_key():
    # Сначала пробуем из session_state, потом из переменных окружения
    return st.session_state.get("deepseek_api_key") or os.getenv("DEEPSEEK_API_KEY", "")

def load_audio(file_path: str, sample_rate: int = 16000, ffmpeg_path: str = "ffmpeg") -> np.ndarray:
    """
    Загружает аудиофайл и конвертирует его в массив numpy с указанной частотой дискретизации.
    
    Args:
        file_path (str): Путь к аудиофайлу (WAV, MP3 и т.д.).
        sample_rate (int): Частота дискретизации (по умолчанию 16000 Гц для Whisper).
        ffmpeg_path (str): Путь к исполняемому файлу ffmpeg (по умолчанию 'ffmpeg' для PATH).
    
    Returns:
        np.ndarray: Аудиоданные в формате float32.
    
    Raises:
        FileNotFoundError: Если аудиофайл или FFmpeg не найдены.
        RuntimeError: Если FFmpeg не может обработать аудио.
    """
    try:
        file_path = Path(file_path).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"Аудиофайл не найден: {file_path}")

        try:
            result = subprocess.run([ffmpeg_path, "-version"], capture_output=True, check=True)
            st.write(f"FFmpeg версия: {result.stdout.decode('utf-8').splitlines()[0]}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"FFmpeg не найден по пути: {ffmpeg_path}. "
                "Убедитесь, что FFmpeg установлен и добавлен в PATH, или укажите правильный путь к ffmpeg.exe."
            )

        stream = ffmpeg.input(str(file_path), threads=0)
        stream = ffmpeg.output(stream, "-", format="s16le", acodec="pcm_s16le", ac=1, ar=sample_rate)
        
        out, err = ffmpeg.run(stream, cmd=[ffmpeg_path, "-nostdin"], capture_stdout=True, capture_stderr=True)

        if not out:
            raise RuntimeError(f"Ошибка FFmpeg: {err.decode('utf-8') if err else 'Неизвестная ошибка'}")

        audio_array = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
        return audio_array

    except Exception as e:
        raise RuntimeError(f"Ошибка при загрузке аудио: {str(e)}")

def transcribe_audio(audio_file: str, model_name: str = "medium", ffmpeg_path: str = "ffmpeg") -> str:
    """
    Транскрибирует аудиофайл с использованием модели Whisper, сохраняя текст в файл.

    Args:
        audio_file (str): Путь к аудиофайлу.
        model_name (str): Название модели Whisper (например, 'base', 'medium', 'large').
        ffmpeg_path (str): Путь к исполняемому файлу ffmpeg.

    Returns:
        str: Распознанный текст.
    """
    try:
        st.write(f"Загрузка модели {model_name}...")
        model = Model(model_name, print_realtime=False, print_progress=True, language="ru")

        st.write(f"Чтение аудиофайла {audio_file}...")
        audio_data = load_audio(audio_file, ffmpeg_path=ffmpeg_path)

        st.write("Начало транскрипции...")
        segments = model.transcribe(audio_data)
        
        full_text = " ".join(segment.text.strip() for segment in segments)
        
        with open("transcription.txt", "w", encoding="utf-8") as f:
            f.write(full_text)
        
        return full_text

    except Exception as e:
        st.error(f"Ошибка при распознавании аудио: {str(e)}")
        return ""

def analyze_sentiment(text: str, model_name: str = "blanchefort/rubert-base-cased-sentiment", max_length: int = 512) -> dict:
    """
    Анализирует тональность текста, разбивая его на части, если он слишком длинный.
    
    Args:
        text (str): Текст для анализа.
        model_name (str): Название модели для анализа тональности.
        max_length (int): Максимальная длина последовательности для модели.
    
    Returns:
        dict: Словарь с меткой (POSITIVE, NEGATIVE, NEUTRAL) и средней уверенностью.
    """
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        tokens = tokenizer.encode(text, add_special_tokens=True)
        num_tokens = len(tokens)
        
        if num_tokens <= max_length:
            result = sentiment_pipeline(text)[0]
            return result
        else:
            chunk_size = max_length - 2  # Учитываем [CLS] и [SEP]
            chunks = []
            for i in range(0, num_tokens, chunk_size):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                if chunk_text.strip():  # Пропускаем пустые фрагменты
                    chunks.append(chunk_text)
            
            if not chunks:
                return {"label": "NEUTRAL", "score": 0.0}
            
            results = sentiment_pipeline(chunks)
            labels = [result['label'] for result in results]
            scores = [result['score'] for result in results]
            
            label_counts = {label: labels.count(label) for label in set(labels)}
            dominant_label = max(label_counts, key=label_counts.get)
            average_score = sum(scores) / len(scores)
            
            return {"label": dominant_label, "score": average_score}
    except Exception as e:
        st.error(f"Ошибка при анализе тональности: {str(e)}")
        return {"label": "ERROR", "score": 0.0}

def get_deepseek_recommendation(transcript: str) -> str:
    """
    Отправляет транскрипцию в Deepseek LLM и возвращает рекомендации по улучшению разговора.
    """
    url = "https://api.deepseek.com/v1/chat/completions"
    api_key = get_deepseek_api_key()
    if not api_key:
        return "API-ключ Deepseek не задан. Пожалуйста, введите его в соответствующее поле."
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "Ты эксперт по коммуникации. Прочитай этот разговор и дай рекомендации, как его улучшить."},
            {"role": "user", "content": transcript}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Ошибка при обращении к Deepseek: {e}"

def main():
    st.title("Распознавание речи, анализ тональности разговора + Рекомендации от ИИ")
    st.write("Загрузите аудиофайл (WAV, MP3), чтобы распознать текст и определить его тональность.")
    
    # Колонка для ввода API-ключа Deepseek
    with st.sidebar:
        st.markdown("### Deepseek API Key")
        api_key_input = st.text_input(
            "Введите ваш Deepseek API ключ:",
            type="password",
            value=st.session_state.get("deepseek_api_key", ""),
            key="deepseek_api_key_input"
        )
        if api_key_input:
            st.session_state["deepseek_api_key"] = api_key_input
        else:
            st.session_state["deepseek_api_key"] = ""
    
    # Загрузка аудиофайла
    uploaded_file = st.file_uploader("Выберите аудиофайл", type=["wav", "mp3"])
    
    # Путь к FFmpeg
    ffmpeg_path = r"C:\Users\Iron_Man\Desktop\Test_APP\.venv\ffmpeg\bin\ffmpeg.exe"  # Укажите правильный путь к ffmpeg.exe
    
    if uploaded_file is not None:
        # Сохранение загруженного файла временно
        temp_dir = "temp_audio"
        os.makedirs(temp_dir, exist_ok=True)
        audio_path = os.path.join(temp_dir, uploaded_file.name)
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio(audio_path)
        
        # Кнопка для запуска анализа
        if st.button("Распознать и проанализировать"):
            with st.spinner("Распознавание аудио..."):
                transcription = transcribe_audio(audio_path, model_name="medium", ffmpeg_path=ffmpeg_path)
            
            if transcription:
                st.subheader("Распознанный текст")
                st.write(transcription)
                
                with st.spinner("Анализ тональности..."):
                    sentiment = analyze_sentiment(transcription)
                
                label_mapping = {
                    "POSITIVE": "Позитивный",
                    "NEGATIVE": "Негативный",
                    "NEUTRAL": "Нейтральный",
                    "ERROR": "Ошибка"
                }
                label = label_mapping.get(sentiment['label'], "Неизвестно")
                score = sentiment['score']
                
                st.subheader("Тональность")
                st.write(f"Тональность: {label} (уверенность: {score:.2%})")
                
                with st.spinner("Генерация рекомендаций от LLM..."):
                    recommendations = get_deepseek_recommendation(transcription)
                st.subheader("Рекомендации по улучшению разговора")
                st.write(recommendations)
                
                # Ссылка на скачивание файла с транскрипцией
                with open("transcription.txt", "r", encoding="utf-8") as f:
                    st.download_button(
                        label="Скачать транскрипцию",
                        data=f,
                        file_name="transcription.txt",
                        mime="text/plain"
                    )
        
        # Очистка временного файла
        if os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    main()