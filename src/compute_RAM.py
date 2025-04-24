import os
import pandas as pd
import torch
import numpy as np
import librosa
import pickle # Для сохранения/загрузки данных из RAM в файл
from tqdm import tqdm # Для отображения прогресса

# Русский алфавит + пробел + пустой символ CTC
RUSSIAN_ALPHABET = "_абвгдеёжзийклмнопрстуфхцчшщъыьэюя "
BLANK_CHAR = '_'
char_map = {char: idx for idx, char in enumerate(RUSSIAN_ALPHABET)}

def text_to_int(text, char_map):
    text = text.lower()
    return [char_map[char] for char in text if char in char_map]

def preprocess_audio(audio_path, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160):
    try:
        audio, orig_sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception as e:
        return None
    
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=8000)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    mean = np.mean(log_mel_spectrogram)
    std = np.std(log_mel_spectrogram)
    
    log_mel_spectrogram = (log_mel_spectrogram - mean) / (std + 1e-6)
    log_mel_spectrogram_tensor = torch.tensor(log_mel_spectrogram, dtype=torch.float32).transpose(0, 1)
    
    return log_mel_spectrogram_tensor

ANNOTATIONS_FILE = "./dataset_target.csv"
AUDIO_DIR = "F:/asr_public_phone_calls_1/0/"
PRECOMPUTED_DATA_PATH = "precomputed_asr_data.pkl" # Файл для сохранения данных

print("Загрузка аннотаций...")
audio_target = pd.read_csv(ANNOTATIONS_FILE)
audio_target.dropna(subset=['text'], inplace=True)
audio_target = audio_target[audio_target['text'].str.strip() != '']
print(f"Найдено {len(audio_target)} записей.")

precomputed_data = [] # Список для хранения пар (спектрограмма, метка)

print("Начало предобработки аудио...")
# Используем tqdm для прогресс-бара
for index, row in tqdm(audio_target.iterrows(), total=len(audio_target)):
    audio_filename = row[0]
    label_text = row[1]
    potential_path_opus = os.path.join(AUDIO_DIR, audio_filename + ".opus")

    if not os.path.exists(potential_path_opus):
        continue

    # Предобработка аудио
    log_mel_spectrogram = preprocess_audio(potential_path_opus)
    if log_mel_spectrogram is None:
        continue

    # Кодирование текста
    label_int = text_to_int(label_text, char_map)
    # Сохраняем как обычный список int, чтобы не хранить лишние тензоры в RAM
    # Добавляем в список. ВАЖНО: сохраняем тензор спектрограммы
    precomputed_data.append((log_mel_spectrogram, label_int))

print(f"\nПредобработано {len(precomputed_data)} аудиофайлов.")

# Оцените размер данных в RAM (приблизительно)
try:
    import sys
    tensor_memory = sum(sys.getsizeof(spec.storage()) for spec, _ in precomputed_data)
    labels_memory = sys.getsizeof(precomputed_data) # Приблизительно для списков/кортежей
    print(f"Примерный объем RAM для спектрограмм: {tensor_memory / (1024**3):.2f} GB")
    print(f"Общий примерный объем данных: {(tensor_memory + labels_memory) / (1024**3):.2f} GB")
except ImportError:
    print("Не удалось оценить размер RAM (нужен модуль sys).")


print(f"Сохранение предобработанных данных в {PRECOMPUTED_DATA_PATH}...")

# Сохраняем список с помощью pickle
with open(PRECOMPUTED_DATA_PATH, 'wb') as f:
    pickle.dump(precomputed_data, f)

print("Предобработка и сохранение завершены.")