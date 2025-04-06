import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from sklearn.model_selection import train_test_split
import time # Для замера времени эпохи

# Русский алфавит + пробел + пустой символ CTC
RUSSIAN_ALPHABET = "_абвгдеёжзийклмнопрстуфхцчшщъыьэюя " # Пробел в конце
char_map = {char: idx for idx, char in enumerate(RUSSIAN_ALPHABET)}
index_map = {idx: char for char, idx in char_map.items()}

# Функции кодирования/декодирования текста
def text_to_int(text, char_map):
    """Преобразует текст в индексы по `char_map`."""
    # Приводим к нижнему регистру и удаляем символы не из алфавита
    text = text.lower()
    return [char_map[char] for char in text if char in char_map]

def int_to_text(indices, index_map):
    """Преобразует индексы обратно в текст, убирая CTC-пустые символы и повторы."""
    text = ""
    last_idx = -1 # Используем -1, чтобы первый символ точно добавился
    for idx in indices:
        if idx == last_idx: # Пропускаем повторы
            continue
        if idx == char_map['_']: # Пропускаем пустой символ CTC (индекс 0)
            last_idx = idx
            continue
        if idx in index_map:
            text += index_map[idx]
        last_idx = idx
    # CTC может генерировать пробелы в начале/конце или двойные пробелы
    text = ' '.join(text.split())
    return text

# Улучшенная обработка аудио
def preprocess_audio(audio_path, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160):
    """Загружает аудио, преобразует в лог-мел-спектрограмму и нормализует."""
    try:
        audio, orig_sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception as e:
        print(f"Ошибка загрузки файла {audio_path}: {e}")
        return None # Возвращаем None, чтобы пропустить в Датасете запись

    # Вычисляем мел-спектрограмму n_fft-окно фурье (25 мс), hop_length-сдвиг окна (10 мс)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=8000)
    # Преобразуем в лог-масштаб (дБ)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Нормализация (по каждому файлу отдельно - instance normalization)
    mean = np.mean(log_mel_spectrogram)     # Среднее
    std = np.std(log_mel_spectrogram)       # Стандартное отклонение
    # Добавляем эпсилон для избежания деления на ноль
    log_mel_spectrogram = (log_mel_spectrogram - mean) / (std + 1e-6)

    # Переводим в тензор PyTorch (T, F)
    log_mel_spectrogram_tensor = torch.tensor(log_mel_spectrogram, dtype=torch.float32).transpose(0, 1)
    return log_mel_spectrogram_tensor

class ASR_CTC_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.15):
        super(ASR_CTC_Model, self).__init__()

        # Добавим сверточные слои для извлечения локальных признаков
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        # Рассчитаем размерность после сверток
        # Изначально (N, 1, T, F) -> (N, 32, T/4, F/4)
        # Нужно преобразовать в (N, T/4, 32 * F/4) для LSTM
        # T - длина последовательности, F - input_dim (n_mels)
        # F_out = input_dim // 4 (из-за stride=(2,2) дважды)
        lstm_input_dim = 32 * (input_dim // 4)

        self.conv_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0 # Dropout между слоями LSTM
        )
        
        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Bidirectional -> *2
        # LogSoftmax применяется в конце

    def forward(self, x):
        # x: (batch, T, F) - вход для LSTM batch_first=True
        # Добавляем канал для Conv2d: (batch, 1, T, F)
        x = x.unsqueeze(1)
        # Применяем сверточные слои
        x = self.conv_layers(x) # -> (batch, 32, T', F') T' = T/4, F' = F/4

        # Изменяем размерность для LSTM: (batch, T', features)
        batch_size, channels, T_prime, F_prime = x.shape
        x = x.permute(0, 2, 1, 3) # -> (batch, T', channels, F')
        x = x.reshape(batch_size, T_prime, channels * F_prime) # -> (batch, T', channels * F')

        # Применяем LSTM
        x, _ = self.lstm(x) # -> (batch, T', hidden_dim * 2)

        # Применяем полносвязный слой
        x = self.fc(x) # -> (batch, T', output_dim)

        # Применяем LogSoftmax для CTC Loss
        # CTC ожидает (T, N, C), где N=batch_size, T=длина посл., C=классы
        x = nn.functional.log_softmax(x, dim=2) # -> (batch, T', output_dim)
        # Permute для CTC Loss будет сделан вне модели
        return x

    def get_output_lengths(self, input_lengths):
        """ Рассчитывает длину выхода после сверточных слоев """
        # Каждый слой Conv2d со stride=2 уменьшает размерность времени в 2 раза
        lengths = input_lengths
        for _ in range(len(self.conv_layers)):
             if isinstance(self.conv_layers[_], nn.Conv2d) and self.conv_layers[_].stride[0] > 1:
                 lengths = torch.div(lengths - 1, self.conv_layers[_].stride[0], rounding_mode='floor') + 1
                 # Более простой вариант для stride=2: lengths = (lengths + 1) // 2
                 #lengths = (lengths + self.conv_layers[_].padding[0] * 2 - self.conv_layers[_].dilation[0] * (self.conv_layers[_].kernel_size[0] - 1) - 1) // self.conv_layers[_].stride[0] + 1

        return lengths

# Датасет 
class CustomAudioDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, char_map):
        self.audio_target = pd.read_csv(annotations_file)

        # Очистка данных: Удаление строк с отсутствующими транскрипциями
        self.audio_target.dropna(subset=['text'], inplace=True)
        self.audio_target = self.audio_target[self.audio_target['text'].str.strip() != '']

        self.audio_dir = audio_dir
        self.char_map = char_map
        print(f"Загружено {len(self.audio_target)} записей после очистки.")


    def __len__(self):
        return len(self.audio_target)

    def __getitem__(self, idx):
        audio_filename = self.audio_target.iloc[idx, 0]
        # Обработка файлов формата .opus
        potential_path_opus = os.path.join(self.audio_dir, audio_filename + ".opus")

        #potential_path_wav = os.path.join(self.audio_dir, audio_filename + ".wav") # Для другого формата

        if os.path.exists(potential_path_opus):
            audio_path = potential_path_opus
        # elif os.path.exists(potential_path_wav):
        #      audio_path = potential_path_wav
        else:
            print(f"Аудиофайл не найден для {audio_filename}")
            # Можно вернуть None или вызвать ошибку, здесь вернем None для пропуска
            return None, None # Пропустим этот элемент в collate_fn

        # Предобработка аудио
        log_mel_spectrogram = preprocess_audio(audio_path)
        if log_mel_spectrogram is None:
            return None, None # Пропускаем, если была ошибка загрузки/обработки

        # Получение и кодирование текста
        label_text = self.audio_target.iloc[idx, 1]
        label_int = text_to_int(label_text, self.char_map)
        label_tensor = torch.tensor(label_int, dtype=torch.long)

        return log_mel_spectrogram, label_tensor

# Функция для сборки батча
def collate_fn_asr(batch):
    # Фильтруем None элементы, которые могли возникнуть из-за ошибок в __getitem__
    batch = [(spec, target) for spec, target in batch if spec is not None]
    if not batch: # Если весь батч оказался пустым
        return None, None, None, None

    inputs, targets = zip(*batch) # Разделяем спектрограммы и целевые тексты

    # Вычисляем длины входов (спектрограмм) до паддинга
    input_lengths = torch.tensor([x.shape[0] for x in inputs], dtype=torch.long)

    # Вычисляем длины выходов (текстов)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

    # Паддинг входов (спектрограмм)
    # Используем 0.0 для паддинга, т.к. данные нормализованы вокруг 0
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0.0)

    # Конкатенация всех целевых последовательностей (уже тензоры)
    # Важно: CTCLoss ожидает targets как один длинный тензор
    targets_concatenated = torch.cat(targets)

    return inputs_padded, targets_concatenated, input_lengths, target_lengths

# Транскрибация аудио
def transcribe_audio(model, path):
    input_tensor = preprocess_audio(path)

    if input_tensor is not None:
        input_tensor = input_tensor.unsqueeze(0).to(device) # Добавляем batch измерение

        with torch.no_grad():
            output = model(input_tensor) # (1, T', output_dim)
            predicted_indices = torch.argmax(output, dim=2).squeeze(0).cpu().tolist()
            decoded_text = int_to_text(predicted_indices, index_map)
            print(f"\nПредсказание для {path}:")
            print(f"  Декодированный текст (жадный поиск): '{decoded_text}'")
    else:
        print(f"Не удалось обработать пример аудио: {path}")

if __name__ == "__main__":
    # Гиперпараметры
    INPUT_DIM = 80                               # n_mels число уровней мэл
    HIDDEN_DIM = 512                             # Число скрытых признаков LSTM
    OUTPUT_DIM = len(RUSSIAN_ALPHABET)
    NUM_LAYERS = 4                               # Число слоев LSTM
    DROPOUT = 0.25                               # Обрубание весов
    BATCH_SIZE = 16                              # Число обрабатываемых аудио за проход
    NUM_EPOCHS = 50                              # Число эпох обучения
    LEARNING_RATE = 1e-4                         # Adam с большими моделями лучше сходится с меньшим LR
    WEIGHT_DECAY = 1e-5                          # Небольшая L2 регуляризация
    CLIP_GRAD_NORM = 5.0                         # Для предотвращения взрыва градиентов
    ANNOTATIONS_FILE = "./dataset_target.csv"    # Путь к CSV с метками датасета
    AUDIO_DIR = "F:/asr_public_phone_calls_1/0/" # Путь к папке с аудио
    MODEL_SAVE_PATH = "asr_ctc_model_best.pth"   # Путь сохранения модели
    TRAIN_SPLIT_RATIO = 0.9                      # 90% на обучение, 10% на валидацию
    PATIENCE_SCHEDULER = 2                       # для ReduceLROnPlateau
    PATIENCE_EARLY_STOPPING = 7                  # остановка после N эпох без улучшений


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Загрузка и разделение данных
    full_dataset = CustomAudioDataset(ANNOTATIONS_FILE, AUDIO_DIR, char_map)

    if len(full_dataset) == 0:
        print("Датасет пуст или не удалось загрузить данные. Выход.")
        exit()

    train_size = int(TRAIN_SPLIT_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Размер обучающей выборки: {len(train_dataset)}")
    print(f"Размер валидационной выборки: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_asr, num_workers=4, pin_memory=True if device == 'cuda' else False)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_asr, num_workers=4, pin_memory=True if device == 'cuda' else False)

    # Инициализация модели
    model = ASR_CTC_Model(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT).to(device)
    print(model) # Вывод структуры модели

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)   
    print(f"Общее количество обучаемых параметров: {total_params:,}")


    # Функция потерь CTC. blank=0 соответствует '_' в алфавите
    ctc_loss = nn.CTCLoss(blank=char_map['_'], reduction='mean', zero_infinity=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # AdamW лучше с weight decay
    
    # Планировщик для уменьшения LR, если loss на валидации не улучшается
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE_SCHEDULER, verbose=True)

    # Цикл обучения и валидации
    best_val_loss = float('inf')
    epochs_no_improve = 0 # Счетчик для ранней остановки

    # for epoch in range(NUM_EPOCHS):
    #     start_time_epoch = time.time()

    #     model.train()
    #     train_loss_accum = 0.0
    #     processed_batches_train = 0

    #     # Прогон по batch-ам
    #     for batch_idx, (inputs, targets, input_lengths, target_lengths) in enumerate(train_dataloader):
    #         # Пропускаем батч, если он пустой (из-за ошибок в collate_fn)
    #         if inputs is None:
    #             print(f"Пропущен пустой батч {batch_idx + 1} в обучении.")
    #             continue

    #         inputs = inputs.to(device)
    #         targets = targets.to(device)
    #         input_lengths = input_lengths.to(device)
    #         target_lengths = target_lengths.to(device)

    #         optimizer.zero_grad()

    #         # Forward pass
    #         outputs = model(inputs) # (batch, T_prime, output_dim)

    #         # Рассчитываем длины выходов после сверток для CTC Loss
    #         output_lengths = model.get_output_lengths(input_lengths).to(device)

    #         # Permute для CTC Loss: (T_prime, batch, output_dim)
    #         log_probs = outputs.permute(1, 0, 2)

    #         # Проверка на валидность длин для CTC Loss. Длина выхода не может быть меньше длины таргета
    #         valid_indices = output_lengths >= target_lengths
    #         if not valid_indices.all():
    #             print(f"Предупреждение: Пропущен батч {batch_idx+1} из-за output_lengths < target_lengths.")
    #             # print("Output lengths:", output_lengths[~valid_indices])
    #             # print("Target lengths:", target_lengths[~valid_indices])
    #             continue # Пропускаем этот батч

    #         # Расчет CTC Loss
    #         try:
    #              loss = ctc_loss(log_probs, targets, output_lengths, target_lengths)
    #         except Exception as e:
    #              print(f"Ошибка в CTC Loss на батче {batch_idx+1}: {e}")
    #              print("log_probs shape:", log_probs.shape)
    #              print("targets shape:", targets.shape)
    #              print("output_lengths:", output_lengths)
    #              print("target_lengths:", target_lengths)
    #              continue # Пропускаем этот батч

    #         # Backward pass и оптимизация
    #         loss.backward()

    #         # Обрезка градиентов
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)

    #         optimizer.step()

    #         train_loss_accum += loss.item()
    #         processed_batches_train += 1

    #         if (batch_idx + 1) % 20 == 0: # Печатаем прогресс каждые 20 батчей
    #             print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx+1}/{len(train_dataloader)}, Train Loss: {loss.item():.4f}")

    #     avg_train_loss = train_loss_accum / processed_batches_train if processed_batches_train > 0 else 0.0

    #     # Валидация
    #     model.eval()
    #     val_loss_accum = 0.0
    #     processed_batches_val = 0
    #     example_predictions = [] # Сохраним несколько примеров

    #     with torch.no_grad():
    #         for batch_idx, (inputs, targets, input_lengths, target_lengths) in enumerate(val_dataloader):
    #             if inputs is None:
    #                 print(f"Пропущен пустой батч {batch_idx + 1} в валидации.")
    #                 continue

    #             inputs = inputs.to(device)
    #             targets = targets.to(device)
    #             input_lengths = input_lengths.to(device)
    #             target_lengths = target_lengths.to(device)

    #             outputs = model(inputs) # (batch, T_prime, output_dim)
    #             output_lengths = model.get_output_lengths(input_lengths).to(device)
    #             log_probs = outputs.permute(1, 0, 2) # (T_prime, batch, output_dim)

    #             valid_indices = output_lengths >= target_lengths
    #             if not valid_indices.all():
    #                print(f"Предупреждение: Пропущен валидационный батч {batch_idx+1} из-за output_lengths < target_lengths.")
    #                continue

    #             try:
    #                 loss = ctc_loss(log_probs, targets, output_lengths, target_lengths)
    #             except Exception as e:
    #                 print(f"Ошибка в CTC Loss на валидационном батче {batch_idx+1}: {e}")
    #                 continue

    #             val_loss_accum += loss.item()
    #             processed_batches_val += 1

    #             # Сохраним примеры из первого валидационного батча
    #             if batch_idx == 0 and len(example_predictions) < 5:
    #                  # Жадное декодирование для примера
    #                  preds = torch.argmax(outputs, dim=2) # (batch, T_prime)
    #                  for i in range(min(len(outputs), 5 - len(example_predictions))):
    #                      pred_indices = preds[i].cpu().tolist()
    #                      decoded_text = int_to_text(pred_indices, index_map)
    #                      # Найти соответствующий таргет для этого примера
    #                      # Таргеты конкатенированы, нужно извлечь нужный сегмент
    #                      start_idx = sum(target_lengths[:i]) #.item()
    #                      end_idx = start_idx + target_lengths[i].item()
    #                      target_indices = targets[start_idx:end_idx].cpu().tolist()
    #                      target_text = int_to_text(target_indices, index_map) # Декодируем для сравнения
    #                      example_predictions.append((decoded_text, target_text))

    #     avg_val_loss = val_loss_accum / processed_batches_val if processed_batches_val > 0 else 0.0
    #     epoch_duration = time.time() - start_time_epoch

    #     print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} Summary:")
    #     print(f"  Duration: {epoch_duration:.2f}s")
    #     print(f"  Avg Train Loss: {avg_train_loss:.4f}")
    #     print(f"  Avg Val Loss: {avg_val_loss:.4f}")
    #     print("  Example Predictions (Predicted | Target):")
    #     for pred, target in example_predictions:
    #         print(f"    - '{pred}' | '{target}'")

    #     # Обновление learning rate
    #     scheduler.step(avg_val_loss)

    #     # Сохранение лучшей модели
    #     if avg_val_loss < best_val_loss:
    #         best_val_loss = avg_val_loss
    #         torch.save(model.state_dict(), MODEL_SAVE_PATH)
    #         print(f"  Validation loss улучшился. Модель сохранена в {MODEL_SAVE_PATH}")
    #         epochs_no_improve = 0 # Сброс счетчика
    #     else:
    #         epochs_no_improve += 1
    #         print(f"  Validation loss не улучшился. Эпох без улучшения: {epochs_no_improve}/{PATIENCE_EARLY_STOPPING}")

    #     if epochs_no_improve >= PATIENCE_EARLY_STOPPING:
    #         print(f"\nРанняя остановка! Validation loss не улучшался {PATIENCE_EARLY_STOPPING} эпох.")
    #         break # Выход из цикла обучения

    #     print("-" * 50)

    # print("Обучение завершено.")

    # Пример использования обученной модели 
    print("\nЗагрузка лучшей модели для предсказания...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    transcribe_audio(model, "C:/Users/danya/Desktop/Record (online-voice-recorder.com).mp3")

    # Используем первые 100 файлов из датасета для примера
    try:
        for index in range(100):
            example_idx_in_full = val_dataset.indices[index] # Индекс в валидационном датасете
            #example_idx_in_full = train_dataset.indices[index] # Индекс в обучающем датасете
            example_audio_filename = full_dataset.audio_target.iloc[example_idx_in_full, 0]
            example_audio_path = os.path.join(AUDIO_DIR, example_audio_filename + ".opus") # или .wav
            #print(f"Пример аудио для предсказания: {example_audio_path}")

            input_tensor = preprocess_audio(example_audio_path)

            if input_tensor is not None:
                input_tensor = input_tensor.unsqueeze(0).to(device) # Добавляем batch измерение

                with torch.no_grad():
                    output = model(input_tensor) # (1, T', output_dim)
                    predicted_indices = torch.argmax(output, dim=2).squeeze(0).cpu().tolist()
                    decoded_text = int_to_text(predicted_indices, index_map)
                    print(f"\nПредсказание для {example_audio_path}:")
                    print(f"  Декодированный текст (жадный поиск): '{decoded_text}'")

                    # Получим реальный текст для сравнения
                    real_text = full_dataset.audio_target.iloc[example_idx_in_full, 1]
                    print(f"  Реальный текст: '{real_text.lower()}'")
            else:
                print(f"Не удалось обработать пример аудио: {example_audio_path}")
        print(1)
    except IndexError:
        print("Не удалось получить пример из валидационного датасета.")
    except FileNotFoundError:
         print(f"Пример аудиофайла не найден: {example_audio_path}")

    # можно добавить Beam Search декодирование с использованием внешних библиотек pyctcdecode, для лучших результатов.