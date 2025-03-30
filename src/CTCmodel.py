import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

# Русский алфавит + пробел (CTC требует пустого символа)
RUSSIAN_ALPHABET = " абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
char_map = {char: idx for idx, char in enumerate(RUSSIAN_ALPHABET)}
index_map = {idx: char for char, idx in char_map.items()}  # Обратный словарь

def text_to_int(text, char_map):
    """Преобразует текст в индексы по `char_map`."""
    return [char_map[char] for char in text if char in char_map]

def int_to_text(indices, index_map):
    """Преобразует индексы обратно в текст, убирая CTC-повторы."""
    text = ""
    prev_char = None
    for idx in indices:
        if idx != prev_char and idx in index_map:
            text += index_map[idx]
        prev_char = idx
    return text.strip()

# CTC модель с обратными связями
class CTCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CTCModel, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Bidirectional -> *2

    def forward(self, x):
        x, _ = self.lstm(x)                     # Выход: (batch, T, hidden_dim * 2)
        x = self.fc(x)                          # Выход: (batch, T, output_dim)
        x = nn.functional.log_softmax(x, dim=2) # Выход: (batch_size, sequence_length, output_dim) вероятности. Обязательно для CTC Loss!
        return x

# Загрузка датасета
class CustomAudioDataset(Dataset):
    def __init__(self, annotations_file, audio_dir):
        self.audio_target = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.max_duration = 15  # Максимальная длительность в секундах
        self.sample_rate = 16000  # Частота дискретизации
        self.max_samples = self.max_duration * self.sample_rate

    def __len__(self):
        return len(self.audio_target)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_target.iloc[idx, 0] + ".opus")
        audio, orig_sr = librosa.load(audio_path, sr=16000, mono=True) # Возможно добавить передискретизацию

        # # Заполнение специальными значениями до 15 секунд
        # if len(audio) < self.max_samples:
        #     pad_length = self.max_samples - len(audio)
        #     audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=-100)

        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=orig_sr, n_mels=80, fmax=8000)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        log_mel_spectrogram = torch.tensor(log_mel_spectrogram, dtype=torch.float32).transpose(0, 1)  # (T, F) формат
        label = self.audio_target.iloc[idx, 1]

        return log_mel_spectrogram, label

# Выравнивание batch до одной длины
def collate_fn(batch):
    inputs, targets = zip(*batch)
    input_lengths = torch.tensor([x.shape[0] for x in inputs], dtype=torch.long)

    # Кодируем текст в индексы
    targets_encoded = [torch.tensor(text_to_int(t, char_map), dtype=torch.long) for t in targets]
    target_lengths = torch.tensor([len(t) for t in targets_encoded], dtype=torch.long)

    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=-100.0) # -100 будет ассоциироваться с тишиной
    targets_padded = torch.cat(targets_encoded)
    return inputs_padded, targets_padded, input_lengths, target_lengths

# Гиперпараметры
input_dim = 80  # Лог-мел-спектрограмма с 80 полосами
hidden_dim = 128
output_dim = len(RUSSIAN_ALPHABET)  # Количество классов (буквы + пробел)
batch_size = 64
num_epochs = 5

# Создание модели
model = CTCModel(input_dim, hidden_dim, output_dim)
ctc_loss = nn.CTCLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Создание датасета и загрузчика данных
dataset = CustomAudioDataset("./dataset_target.csv", "F:/asr_public_phone_calls_1/0/")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Цикл обучения
for epoch in range(num_epochs):
    total_batches = len(dataloader)
    for batch_idx, (inputs, targets, input_lengths, target_lengths) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = ctc_loss(outputs.permute(1, 0, 2), targets, input_lengths, target_lengths)
        loss.backward()
        
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{total_batches}, Loss: {loss.item()}")

    # Вывод предсказаний для 10 случайных аудио
    model.eval()
    with torch.no_grad():
        print(f"Epoch {epoch+1} - Предсказания для 10 аудио:")
        for i in range(10):
            input_spec = inputs[i].unsqueeze(0)  # Извлекаем одно аудио для предсказания
            output = model(input_spec)
            predicted_indices = torch.argmax(output, dim=2).squeeze(0).tolist()
            decoded_text = int_to_text(predicted_indices, index_map)
            target_text = targets[i]
            print(f"Предсказание: {decoded_text}, Целевой текст: {target_text}")
    model.train()

torch.save(model.state_dict(), f"ctc_model.pth")

# Пример декодирования предсказаний
checkpoint_path = "ctc_model.pth"  # Указать путь к нужному чекпоинту
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print("Модель загружена из", checkpoint_path)

# audio, orig_sr = librosa.load("F:/asr_public_phone_calls_1/0/4dec5847aec3.opus", sr=16000, mono=True) # Возможно добавить передискретизацию
# mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=orig_sr, n_mels=80, fmax=8000)
# log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
# log_mel_spectrogram = torch.tensor(log_mel_spectrogram, dtype=torch.float32).transpose(0, 1)  # (T, F) формат


# outputs = model(log_mel_spectrogram)  # (N, T, C)
# outputs = outputs.permute(1, 0, 2)  # Преобразуем в (T, N, C)
# predicted_indices = torch.argmax(outputs, dim=2).transpose(0, 1)  # (N, T)
# decoded_text = [int_to_text(seq.tolist(), index_map) for seq in predicted_indices]
# print(decoded_text)

# outputs = model(log_mel_spectrogram)
# predicted_indices = torch.argmax(outputs, dim=2)  # Получаем индексы символов
# decoded_text = [int_to_text(seq.tolist(), index_map) for seq in predicted_indices]
# print(decoded_text)
# print(1)