import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset # Убрали random_split, т.к. делаем вручную
from torch.nn.utils.rnn import pad_sequence
import librosa
import numpy as np
import time
from tqdm import tqdm
import jiwer

# Добавлены импорты для TensorBoard и Аугментации
from torch.utils.tensorboard import SummaryWriter
import torchaudio.transforms as T

# Русский алфавит + пробел + пустой символ CTC
RUSSIAN_ALPHABET = "_абвгдеёжзийклмнопрстуфхцчшщъыьэюя "
BLANK_CHAR = '_'
char_map = {char: idx for idx, char in enumerate(RUSSIAN_ALPHABET)}
index_map = {idx: char for char, idx in char_map.items()}

# Функции кодирования/декодирования текста (без изменений)
def text_to_int(text, char_map):
    """Преобразует текст в индексы по `char_map`."""

    text = text.lower()
    # Пропускаем символы, которых нет в карте, чтобы избежать ошибок
    return [char_map[char] for char in text if char in char_map]

def int_to_text(indices, index_map, blank_char=BLANK_CHAR):
    """Преобразует индексы обратно в текст, убирая CTC-пустые символы и повторы."""

    text = ""
    blank_idx = char_map.get(blank_char, -1)
    last_idx = -1
    for idx in indices:
        if idx == last_idx: continue
        if idx == blank_idx:
            last_idx = idx
            continue
        # Проверяем наличие индекса в карте перед добавлением
        if idx in index_map:
            text += index_map[idx]
        last_idx = idx
    text = ' '.join(text.split())
    return text

# Улучшенная обработка аудио (возвращает F, T)
def preprocess_audio(audio_path, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160):
    """Загружает аудио, преобразует в лог-мел-спектрограмму и нормализует."""

    try:
        audio, orig_sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception as e:
        print(f"Ошибка загрузки файла {audio_path}: {e}")
        return None
    
    # Дополнение слишком коротких аудио, которые могут вызвать ошибки в Conv слоях
    if len(audio) < n_fft: # Если аудио короче одного окна FFT
        padding_needed = n_fft - len(audio)
        # Дополняем нулями (тишиной) в конец
        audio = np.pad(audio, (0, padding_needed), mode='constant', constant_values=0)
 
    # Вычисляем мел-спектрограмму
    try:
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=8000)
    except Exception as e:
         print(f"Ошибка при вычислении melspectrogram для {audio_path} (даже после паддинга): {e}")
         return None    # Проверка на нулевую энергию (тишину), которая может дать -inf после power_to_db
    
    if np.max(mel_spectrogram) == 0:
         # print(f"Предупреждение: Нулевая энергия в файле {audio_path}, пропускаем.")
         return None 
    
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Нормализация
    mean = np.mean(log_mel_spectrogram)
    std = np.std(log_mel_spectrogram)

    # Проверка на нулевое стандартное отклонение (монотонный сигнал)
    if std < 1e-6:
        # print(f"Предупреждение: Нулевое стд. отклонение в файле {audio_path}, пропускаем.")
        return None
    
    log_mel_spectrogram = (log_mel_spectrogram - mean) / (std + 1e-6)

    log_mel_spectrogram_tensor = torch.tensor(log_mel_spectrogram, dtype=torch.float32) # (F, T)
    return log_mel_spectrogram_tensor

# ASR_CTC_Model 
class ASR_CTC_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.15):
        super(ASR_CTC_Model, self).__init__()
        # Слои Conv2d ожидают вход (B, C, H, W) или (B, C, F, T)
        self.conv_layers = nn.Sequential(
            # Вход: (B, 1, F=input_dim, T)
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), # -> (B, 32, F/2, T/2)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), # -> (B, 32, F/4, T/4)
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
    
        # Расчет входной размерности для LSTM
        lstm_input_dim = 32 * (input_dim // 4) # Каналы * (F уменьшается в 2 раза дважды (из-за stride[0]=2))

        # Dropout слой
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

        self.fc = nn.Linear(hidden_dim * 2, output_dim) # Bidirectional -> *2

    def forward(self, x):
        # x: (batch, F, T) - входной формат от DataLoader
        x = x.unsqueeze(1) # -> (B, 1, F, T) - добавляем канал для Conv2d
        x = self.conv_layers(x) # -> (B, 32, F/4, T/4)

        # Изменяем размерность для LSTM: (batch, T', features)
        batch_size, channels, F_prime, T_prime = x.shape
        x = x.permute(0, 3, 1, 2) # -> (B, T', C, F') -> T'=T/4, F'=F/4
        x = x.reshape(batch_size, T_prime, channels * F_prime) # -> (B, T', 32*F/4)

        #  Dropout после сверток
        x = self.conv_dropout(x)

        x, _ = self.lstm(x) # -> (B, T', H*2)

        # Dropout перед FC
        x = self.fc_dropout(x)

        x = self.fc(x) # -> (B, T', Output)

        # Применяем LogSoftmax для CTC Loss
        # CTC ожидает (T, N, C), где N=batch_size, T=длина посл., C=классы
        x = nn.functional.log_softmax(x, dim=2) # -> (B, T', Output)
        return x

    def get_output_lengths(self, input_lengths_T): # Принимает длины по оси T
        """ Рассчитывает длину ВЫХОДА по оси T после сверток """
        lengths = input_lengths_T

        # Уменьшение по оси T определяется stride[1] для Conv2d при входе (B, C, F, T)
        
        # Первый Conv: stride=(2, 2) -> stride[1]=2
        if isinstance(self.conv_layers[0], nn.Conv2d) and self.conv_layers[0].stride[1] > 1:
             lengths = torch.div(lengths - 1, self.conv_layers[0].stride[1], rounding_mode='floor') + 1
        
        # Второй Conv: stride=(2, 2) -> stride[1]=2
        if isinstance(self.conv_layers[3], nn.Conv2d) and self.conv_layers[3].stride[1] > 1:
             lengths = torch.div(lengths - 1, self.conv_layers[3].stride[1], rounding_mode='floor') + 1

        # Итого T_out = T_in / 4
        return lengths

# Dataset для АУГМЕНТАЦИИ и возврата текста
class CustomAudioDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, char_map, apply_augmentation=False, freq_mask_param=27, time_mask_param=70):
        self.audio_target = pd.read_csv(annotations_file)

        self.audio_target.dropna(subset=['text'], inplace=True)
        self.audio_target = self.audio_target[self.audio_target['text'].str.strip() != '']

        self.audio_dir = audio_dir
        self.char_map = char_map
        self.apply_augmentation = apply_augmentation

        # Аугментация
        if self.apply_augmentation:
            self.spec_augment = nn.Sequential(
                T.FrequencyMasking(freq_mask_param=freq_mask_param),
                T.TimeMasking(time_mask_param=time_mask_param)
            ).eval() # Переводим в eval(), чтобы dropout внутри не работал, если он там есть
            print(f"Аугментация SpecAugment ({freq_mask_param=}, {time_mask_param=}) включена для этого датасета.")
        else:
            self.spec_augment = None
            print("Аугментация SpecAugment выключена для этого датасета.")

        print(f"Загружено {len(self.audio_target)} записей после очистки.")

    def __len__(self):
        return len(self.audio_target)

    def __getitem__(self, idx):
        if idx >= len(self.audio_target): # Проверка индекса
            raise IndexError("Индекс за пределами датасета")
        
        audio_filename = self.audio_target.iloc[idx, 0]
        potential_path_opus = os.path.join(self.audio_dir, audio_filename + ".opus")

        if not os.path.exists(potential_path_opus):
            # print(f"Аудиофайл не найден для {audio_filename}")
            return None, None, None

        # Предобработка возвращает (F, T) или None
        log_mel_spectrogram = preprocess_audio(potential_path_opus)
        if log_mel_spectrogram is None:
            return None, None, None

        # Применяем аугментацию, если нужно
        if self.apply_augmentation and self.spec_augment is not None:
            try:
                log_mel_spectrogram = self.spec_augment(log_mel_spectrogram)
            except Exception as e:
                print(f"Ошибка аугментации файла {audio_filename}: {e}")
                return None, None, None # Пропускаем при ошибке аугментации

        label_text = self.audio_target.iloc[idx, 1]
        # Проверка на NaN или float перед кодированием текста
        if not isinstance(label_text, str):
             # print(f"Предупреждение: Некорректная метка '{label_text}' для файла {audio_filename}, пропускаем.")
             return None, None, None
        label_int = text_to_int(label_text, self.char_map)
        label_tensor = torch.tensor(label_int, dtype=torch.long)

        # Возвращаем 3 элемента
        return log_mel_spectrogram, label_tensor, label_text.lower()

# collate_fn для обработки формата (F, T) и 5 элементов
def collate_fn_asr_wer(batch):
    # Фильтруем None (кортежи из 3-х элементов)
    batch = [(spec, target_tensor, target_text)
             for spec, target_tensor, target_text in batch
             if spec is not None and target_tensor is not None and target_text is not None]
    if not batch:
        return None, None, None, None, None

    inputs_feat_time, targets_tensor, targets_text = zip(*batch) # inputs в формате (F, T)

    # Длины по ВРЕМЕННОЙ оси (T) - вторая размерность
    input_lengths = torch.tensor([x.shape[1] for x in inputs_feat_time], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets_tensor], dtype=torch.long)

    # Паддинг входов (спектрограмм) по временной оси T
    try:
        inputs_permuted = [x.permute(1, 0) for x in inputs_feat_time] # список (T, F)
        inputs_padded_permuted = pad_sequence(inputs_permuted, batch_first=True, padding_value=0.0) # (B, T_padded, F)
        inputs_padded = inputs_padded_permuted.permute(0, 2, 1) # (B, F, T_padded)
    except Exception as e:
        print(f"Ошибка паддинга в collate_fn: {e}")
        # Попытка вывести размеры для отладки
        for i, spec in enumerate(inputs_feat_time):
            print(f"  Элемент {i} форма: {spec.shape}")
        return None, None, None, None, None # Пропускаем батч при ошибке

    targets_concatenated = torch.cat(targets_tensor)

    return inputs_padded, targets_concatenated, input_lengths, target_lengths, list(targets_text)

# Расчет WER
@torch.no_grad()
def evaluate_wer(model: nn.Module,
                 dataloader: DataLoader,
                 device: torch.device,
                 index_map: dict) -> tuple[float, list, list]:
    
    model.eval()
    all_predictions = []
    all_references = []

    print(f"\nНачало оценки WER на устройстве {device}...")

    for batch_data in tqdm(dataloader, desc="WER Evaluation"):
        if batch_data is None or batch_data[0] is None: continue
        try:
            inputs, _, input_lengths_T, _, ref_texts = batch_data # Длины по оси T
        except ValueError as e:
            print(f"Ошибка распаковки батча в evaluate_wer: {e}")
            print(f"Получено элементов: {len(batch_data) if isinstance(batch_data, (list, tuple)) else 'Не список/кортеж'}")
            continue

        inputs = inputs.to(device)
        input_lengths_T_dev = input_lengths_T.to(device)

        try:
            outputs = model(inputs) # (B, T_out, C)
            output_lengths = model.get_output_lengths(input_lengths_T_dev) # (B)
        except Exception as e:
            print(f"Ошибка model forward/get_output_lengths в evaluate_wer: {e}")
            # Попытка вывести размеры для отладки
            print(f"  Форма inputs: {inputs.shape}, Длины входа T: {input_lengths_T_dev}")
            continue

        preds_indices = torch.argmax(outputs, dim=2) # (B, T_out)
        preds_indices_cpu = preds_indices.cpu().tolist()
        output_lengths_cpu = output_lengths.cpu().tolist()

        for i in range(len(inputs)):
            actual_len = output_lengths_cpu[i]
            if actual_len <= 0 :
                pred_text = ""
            else:
                pred_text = int_to_text(preds_indices_cpu[i][:actual_len], index_map)
            ref_text = ref_texts[i]
            all_predictions.append(pred_text)
            all_references.append(ref_text)

    calculated_wer = float('inf')
    if all_references and all_predictions:
        if len(all_references) != len(all_predictions):
             print(f"Предупреждение: Разная длина референсов ({len(all_references)}) и предсказаний ({len(all_predictions)})!")
        else:
            print("Расчет WER...")
            try:
                calculated_wer = jiwer.wer(all_references, all_predictions) * 100
                print(f"Ошибка WER = {calculated_wer:.2f}%")
            except Exception as e:
                print(f"Ошибка при расчете WER с помощью jiwer: {e}")
    else:
        print("Предупреждение: Не удалось собрать данные для расчета WER.")

    # model.train() # Убрали, пусть вызывающий код решает, когда переключать режим
    return calculated_wer, all_references, all_predictions


if __name__ == "__main__":
    SUBSET_FRACTION = 0.001

    # Гиперпараметры
    INPUT_DIM = 80                                          # n_mels число уровней мэл
    HIDDEN_DIM = 512                                        # Число скрытых признаков LSTM
    OUTPUT_DIM = len(RUSSIAN_ALPHABET)
    NUM_LAYERS = 4                                          # Число слоев LSTM
    DROPOUT = 0.3                                           # Обрубание весов
    BATCH_SIZE = 16                                         # Число обрабатываемых аудио за проход
    NUM_EPOCHS = 50                                         # Число эпох обучения
    LEARNING_RATE = 5e-5                                    # Adam с большими моделями лучше сходится с меньшим LR
    WEIGHT_DECAY = 1e-4                                     # Небольшая L2 регуляризация
    CLIP_GRAD_NORM = 5.0                                    # Для предотвращения взрыва градиентов
    ANNOTATIONS_FILE = "./dataset_target.csv"               # Путь к CSV с метками датасета
    AUDIO_DIR = "F:/asr_public_phone_calls_1/0/"            # Путь к папке с аудио
    MODEL_SAVE_PATH = "asr_ctc_model_best_v3_aug_log.pth"   # Путь сохранения модели
    MODEL_PATH = "./asr_ctc_model_best_6epoch.pth"          # Путь для дообучения модели
    TRAIN_SPLIT_RATIO = 0.9                                 # 90% на обучение, 10% на валидацию
    PATIENCE_SCHEDULER = 3                                  # для ReduceLROnPlateau
    PATIENCE_EARLY_STOPPING = 10                            # Число эпох без улучшения для выхода
    FREQ_MASK_PARAM = 27                                    # Аугментация по частоте
    TIME_MASK_PARAM = 70                                    # Аугментация по времени

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Инициализация TensorBoard
    run_name = f"asr_lr{LEARNING_RATE}_wd{WEIGHT_DECAY}_dr{DROPOUT}_aug"
    log_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(log_dir)
    print(f"Логи TensorBoard: {log_dir}")

    # Загрузка и разделение данных
    print("Создание датасетов...")
    full_dataset_train_version = CustomAudioDataset(ANNOTATIONS_FILE, AUDIO_DIR, char_map, apply_augmentation=True, freq_mask_param=FREQ_MASK_PARAM, time_mask_param=TIME_MASK_PARAM)
    full_dataset_val_version = CustomAudioDataset(ANNOTATIONS_FILE, AUDIO_DIR, char_map, apply_augmentation=False)

    if len(full_dataset_train_version) == 0: exit()

    dataset_size = len(full_dataset_train_version)

    # УДАЛИТЬ для 0.01 датасета
    # subset_size = int(dataset_size * SUBSET_FRACTION)
    # subset_indices = list(range(subset_size)) # Берем первые N индексов
    # sub_train_size = int(len(subset_indices) * TRAIN_SPLIT_RATIO)
    # sub_val_size = len(subset_indices) - sub_train_size
    # train_indices = subset_indices[:sub_train_size]
    # val_indices = subset_indices[sub_train_size:]
    # train_subset_aug = Subset(full_dataset_train_version, train_indices)
    # val_subset_noaug = Subset(full_dataset_val_version, val_indices)

    train_size = int(TRAIN_SPLIT_RATIO * dataset_size)
    val_size = dataset_size - train_size
    indices = list(range(dataset_size))

    #Фиксированное разбиение
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset_aug = Subset(full_dataset_train_version, train_indices)
    val_subset_noaug = Subset(full_dataset_val_version, val_indices)

    print(f"Размер обучающей выборки: {len(train_subset_aug)}")
    print(f"Размер валидационной выборки: {len(val_subset_noaug)}")

    # DataLoaders
    train_dataloader = DataLoader(train_subset_aug, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_asr_wer, num_workers=4, pin_memory=True if device == 'cuda' else False, persistent_workers=True if torch.cuda.is_available() and 4 > 0 else False)
    val_dataloader = DataLoader(val_subset_noaug, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_asr_wer, num_workers=4, pin_memory=True if device == 'cuda' else False, persistent_workers=True if torch.cuda.is_available() and 4 > 0 else False)

    # Инициализация модели
    model = ASR_CTC_Model(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT).to(device)

    # Загрузка весов, если нужно продолжить обучение
    # model_load_path = "путь/к/предыдущей/модели.pth"
    # if os.path.exists(model_load_path):
    #     print(f"Загрузка весов из {model_load_path}")
    #     model.load_state_dict(torch.load(model_load_path, map_location=device))

    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Общее количество обучаемых параметров: {total_params:,}")

    # Loss, Optimizer, Scheduler
    ctc_loss = nn.CTCLoss(blank=char_map[BLANK_CHAR], reduction='mean', zero_infinity=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE_SCHEDULER, verbose=True) # mode='min' для WER

    # Цикл обучения
    best_val_wer = float('inf')
    epochs_no_improve = 0
    start_epoch = 0 # Для продолжения обучения

    print(f"\n Начало обучения ({NUM_EPOCHS} эпох) ")
    for epoch in range(start_epoch, NUM_EPOCHS):
        start_time_epoch = time.time()
        # Обучение 
        model.train()
        train_loss_accum = 0.0
        processed_samples_train = 0

        for batch_idx, batch_data in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Training")):
            if batch_data is None or batch_data[0] is None: continue

            try:
                inputs, targets_concat, input_lengths_T, target_lengths, _ = batch_data
            except ValueError as e:
                print(f"\nОшибка распаковки трен. батча {batch_idx+1}: {e}")
                continue

            # Проверка на пустые тензоры после фильтрации в collate
            if inputs.numel() == 0 or targets_concat.numel() == 0:
                 # print(f"\nПропущен пустой батч {batch_idx+1} после collate.")
                 continue

            inputs = inputs.to(device, non_blocking=True)
            targets = targets_concat.to(device, non_blocking=True)
            input_lengths = input_lengths_T.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            optimizer.zero_grad()

            try:
                outputs = model(inputs) # (B, T_out, C)
                output_lengths = model.get_output_lengths(input_lengths).to(device) # (B)

                # Проверка валидности длин
                valid_indices = output_lengths >= target_lengths
                valid_indices &= (output_lengths > 0) & (target_lengths > 0)

                if not valid_indices.any(): # Если нет валидных в батче
                    # print(f"\nПропущен батч {batch_idx+1}: нет валидных длин.")
                    continue

                # Фильтруем батч для лосса, если не все валидны
                if not valid_indices.all():
                    # print(f"\nФильтрация батча {batch_idx+1} для лосса...")
                    outputs = outputs[valid_indices]
                    targets_list = []
                    current_pos = 0
                    valid_target_lengths_list = []
                    target_lengths_cpu = target_lengths.cpu() # Для итерации
                    targets_cpu = targets.cpu() # Для срезов

                    for i in range(len(target_lengths_cpu)):
                        target_len_i = target_lengths_cpu[i].item()
                        if valid_indices[i]:
                             targets_list.append(targets_cpu[current_pos : current_pos + target_len_i])
                             valid_target_lengths_list.append(target_len_i)
                        current_pos += target_len_i

                    targets = torch.cat(targets_list).to(device, non_blocking=True)
                    output_lengths = output_lengths[valid_indices]
                    target_lengths = torch.tensor(valid_target_lengths_list, dtype=torch.long, device=device)

                # Если после фильтрации ничего не осталось
                if targets.numel() == 0:
                    continue

                log_probs = outputs.permute(1, 0, 2) # (T_out, B_filtered, C)

                loss = ctc_loss(log_probs, targets, output_lengths, target_lengths)
                if torch.isnan(loss) or torch.isinf(loss):
                     print(f"\nNaN/Inf loss на батче {batch_idx+1}, пропуск.")
                     continue

            except Exception as e:
                 print(f"\nОшибка forward/loss на трен. батче {batch_idx+1}: {e}")
                 continue

            # Backward и Step
            try:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                optimizer.step()
            except Exception as e:
                print(f"\nОшибка backward/step на батче {batch_idx+1}: {e}")
                optimizer.zero_grad() # Сбрасываем градиенты при ошибке
                continue

            # Успешный шаг
            batch_size_processed = outputs.size(0) # Реальный размер батча после фильтрации
            train_loss_accum += loss.item() * batch_size_processed
            processed_samples_train += batch_size_processed

            # Логирование батча
            if (batch_idx + 1) % 100 == 0: # Логируем реже
                 global_step = epoch * len(train_dataloader) + batch_idx
                 writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                 # print(f"DEBUG: Logged train_batch loss at step {global_step}") # Отладочный вывод

        avg_train_loss = train_loss_accum / processed_samples_train if processed_samples_train > 0 else float('inf')

        # Валидация 
        # Переключаем модель в eval режим ВНУТРИ evaluate_wer
        epoch_wer, val_refs, val_preds = evaluate_wer(model, val_dataloader, device, index_map)
        # evaluate_wer возвращает модель в режим train в конце

        epoch_duration = time.time() - start_time_epoch

        # Логирование эпохи
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch + 1) # Логируем по номеру эпохи (начиная с 1)
        writer.add_scalar('WER/validation', epoch_wer, epoch + 1)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch + 1)
        writer.add_scalar('EpochDuration_sec', epoch_duration, epoch + 1)

        # Вывод результатов эпохи
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} Summary:")
        print(f"  Duration: {epoch_duration:.2f}s")
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        print(f"  Validation WER: {epoch_wer:.2f}%")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.3e}")
        print("  Example Predictions (Predicted | Target):")
        for i in range(min(5, len(val_preds))):
            print(f"    - '{val_preds[i]}' | '{val_refs[i]}'")

        # Обновление планировщика, сохранение лучшей модели, ранняя остановка
        scheduler.step(epoch_wer)

        if epoch_wer < best_val_wer:
            best_val_wer = epoch_wer
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  Validation WER улучшился до {best_val_wer:.2f}%. Модель сохранена.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  Validation WER не улучшился. Эпох без улучшения: {epochs_no_improve}/{PATIENCE_EARLY_STOPPING}")

        if epochs_no_improve >= PATIENCE_EARLY_STOPPING:
            print(f"\nРанняя остановка на эпохе {epoch+1}! Validation WER не улучшался {PATIENCE_EARLY_STOPPING} эпох.")
            break

        print("-" * 50)

    writer.close()
    print(f"Обучение завершено. Лучший Validation WER: {best_val_wer:.2f}% сохранен в {MODEL_SAVE_PATH}")

    # Финальная оценка (загружаем лучшую модель)
    print("\n Финальная оценка лучшей модели")
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f"Модель {MODEL_SAVE_PATH} успешно загружена.")
        final_wer, final_refs, final_preds = evaluate_wer(model, val_dataloader, device, index_map)

        if final_wer != float('inf'):
            print(f"\nИтоговый WER лучшей модели на валидационной выборке: {final_wer:.2f}%")
            measures = jiwer.compute_measures(final_refs, final_preds)
            print(f"MER: {measures['mer']:.4f}, WIL: {measures['wil']:.4f}")
            print(f"Вставки: {measures['insertions']}, Удаления: {measures['deletions']}, Замены: {measures['substitutions']}")
            # Сохранение результатов
            results_df = pd.DataFrame({'reference': final_refs, 'prediction': final_preds})
            results_df.to_csv('final_wer_results.csv', index=False, encoding='utf-8')
            print("Детальные результаты сохранены в final_wer_results.csv")
        else:
            print("Не удалось рассчитать финальный WER.")
    except FileNotFoundError:
        print(f"Не найден файл лучшей модели: {MODEL_SAVE_PATH}.")
    except Exception as e:
        print(f"Ошибка при финальной оценке: {e}")
        import traceback
        traceback.print_exc()