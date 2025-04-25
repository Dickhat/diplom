import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, random_split
from torch.nn.utils.rnn import pad_sequence
import librosa
import numpy as np
import time
from tqdm import tqdm
import jiwer

# Добавлены импорты для TensorBoard и Аугментации
from torch.utils.tensorboard import SummaryWriter
import torchaudio.transforms as T

# Для параллельных вычислений
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.data.distributed import DistributedSampler

import gc

# Русский алфавит + пробел + пустой символ CTC
RUSSIAN_ALPHABET = "_абвгдеёжзийклмнопрстуфхцчшщъыьэюя "
BLANK_CHAR = '_'
char_map = {char: idx for idx, char in enumerate(RUSSIAN_ALPHABET)}
index_map = {idx: char for char, idx in char_map.items()}

# Функции кодирования/декодирования текста
def text_to_int(text, char_map):
    """Преобразует текст в индексы по `char_map`."""

    text = text.lower()
    # Пропускаем символы, которых нет в map, чтобы избежать ошибок
    return [char_map[char] for char in text if char in char_map]

def int_to_text(indices, index_map, blank_char=BLANK_CHAR):

    """Преобразует индексы обратно в текст, убирая CTC-пустые символы и повторы."""

    text = ""
    blank_idx = char_map.get(blank_char, -1)
    last_idx = -1
    for idx in indices:
        if idx == last_idx: 
            continue
        if idx == blank_idx:
            last_idx = idx
            continue
        # Проверяем наличие индекса в map перед добавлением
        if idx in index_map:
            text += index_map[idx]
        last_idx = idx
    text = ' '.join(text.split())
    return text

# Обработка аудио (возвращает лог-мел-спектрограмму F, T)
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
         print(f"Ошибка при вычислении melspectrogram для {audio_path}: {e}")
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
            # Вход: (B, C=1, F=input_dim, T=time)
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), # -> (B, 32, F/2, T/2)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), # -> (B, 32, F/4, T/4)
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
    
        # Расчет входной размерности для LSTM
        lstm_input_dim = 32 * (input_dim // 4) # Каналы * (F уменьшается в 2 раза дважды)

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


class SingleSourceAudioDataset(Dataset):
    def __init__(self, annotation_file, audio_dir, char_map, apply_augmentation=False, freq_mask_param=27, time_mask_param=70):
        """ Работает с ОДНИМ файлом аннотации и ОДНОЙ аудио директорией. """
        self.audio_dir = audio_dir
        self.char_map = char_map
        self.apply_augmentation = apply_augmentation
        self.spec_augment = None

        try:
            self.audio_target = pd.read_csv(annotation_file)
            self.audio_target.dropna(subset=['text'], inplace=True)
            self.audio_target = self.audio_target[self.audio_target['text'].apply(lambda x: isinstance(x, str))]
            self.audio_target = self.audio_target[self.audio_target['text'].str.strip() != '']
            self.audio_target.reset_index(drop=True, inplace=True) # Важно для Subset/ConcatDataset
        except FileNotFoundError:
            print(f"Ошибка: Файл аннотации не найден: {annotation_file}")
            self.audio_target = pd.DataFrame(columns=['filename', 'text']) # Пустой
        except Exception as e:
            print(f"Ошибка при чтении {annotation_file}: {e}")
            self.audio_target = pd.DataFrame(columns=['filename', 'text'])

        if self.apply_augmentation and len(self.audio_target) > 0:
            self.spec_augment = nn.Sequential(T.FrequencyMasking(freq_mask_param=freq_mask_param),T.TimeMasking(time_mask_param=time_mask_param)).eval()

    def __len__(self):
        return len(self.audio_target)

    def __getitem__(self, idx):
        if idx >= len(self.audio_target):
            raise IndexError(f"Индекс {idx} за пределами датасета (размер {len(self.audio_target)})")
        try:
            row = self.audio_target.iloc[idx]
            audio_filename_no_ext = row['filename']
            label_text = row['text']

            potential_path_opus = os.path.join(self.audio_dir, str(audio_filename_no_ext) + ".opus")
            
            if not os.path.exists(potential_path_opus): 
                return None, None, None
           
            log_mel_spectrogram = preprocess_audio(potential_path_opus)
            
            if log_mel_spectrogram is None: 
                return None, None, None
            
            if self.apply_augmentation and self.spec_augment is not None:
                log_mel_spectrogram = self.spec_augment(log_mel_spectrogram)

            if not isinstance(label_text, str): 
                return None, None, None
            
            label_int = text_to_int(label_text, self.char_map) 
            label_tensor = torch.tensor(label_int, dtype=torch.long)

            return log_mel_spectrogram, label_tensor, label_text.lower()
        except Exception as e: 
            print(f"Ошибка в __getitem__ (idx={idx}): {e}"); return None, None, None

# Dataset для АУГМЕНТАЦИИ и возврата текста (УДАЛИТЬ)
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

# collate_fn для обработки формата (F, T) и преобразования в 5 элементов
def collate_fn_asr_wer(batch):
    # Фильтруем None (кортежи из 3-х элементов)
    batch = [(spec, target_tensor, target_text)
             for spec, target_tensor, target_text in batch
             if spec is not None and target_tensor is not None and target_text is not None]
    if not batch:
        return None, None, None, None, None

    # batch = список кортежей [(s1, t1, tx1), (s2, t2, tx2)] -> Итератор (s1, s2), (t1, t2), (tx1, tx2)
    inputs_feat_time, targets_tensor, targets_text = zip(*batch) # inputs в формате (F, T), остальные одномерные

    # Длины по ВРЕМЕННОЙ оси (T) - вторая размерность
    input_lengths = torch.tensor([x.shape[1] for x in inputs_feat_time], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets_tensor], dtype=torch.long)

    # Паддинг входов (спектрограмм) по временной оси T, чтобы были одинаковой длины
    try:
        inputs_permuted = [x.permute(1, 0) for x in inputs_feat_time] # список (T, F)
        inputs_padded_permuted = pad_sequence(inputs_permuted, batch_first=True, padding_value=0.0) # (B, T_padded, F)
        inputs_padded = inputs_padded_permuted.permute(0, 2, 1) # (B, F, T_padded)
    except Exception as e:
        print(f"Ошибка паддинга в collate_fn: {e}")
        return None, None, None, None, None # Пропускаем батч при ошибке

    targets_concatenated = torch.cat(targets_tensor)

    # inputs_padded - Спектрограммы (одинакого размера),
    # targets_concatenated - целевой результат,
    # input_lengths - исходная длина спектрограмм,
    # target_lengths - исходная длина целевого результата,
    # list(targets_text) - целевой текст
    return inputs_padded, targets_concatenated, input_lengths, target_lengths, list(targets_text)

# Расчет WER
@torch.no_grad()
def evaluate_wer(model: nn.Module,
                 dataloader: DataLoader,
                 device: torch.device,
                 index_map: dict,
                 ctc_loss: nn.CTCLoss):
    
    model.eval()

    val_loss_accum_gpu = 0.0
    processed_samples_val_gpu = 0

    local_predictions = []
    local_references = []

    #print(f"\nНачало оценки WER на устройстве {device}...")
    val_pbar = tqdm(dataloader, desc="WER Evaluation", disable=not is_master)

    for batch_data in val_pbar:
        if batch_data is None or batch_data[0] is None: 
            continue
        try:
            inputs, targets_concat, input_lengths_T, target_lengths_, ref_texts = batch_data # Длины по оси T
        except ValueError as e:
            print(f"Ошибка распаковки батча в evaluate_wer: {e}")
            print(f"Получено элементов: {len(batch_data) if isinstance(batch_data, (list, tuple)) else 'Не список/кортеж'}")
            continue

        inputs = inputs.to(device, non_blocking=True)
        targets = targets_concat.to(device, non_blocking=True)
        input_lengths = input_lengths_T.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)
        batch_size = inputs.size(0)

        try:
            outputs = model(inputs) # (B, T_out, C)
            output_lengths = model.module.get_output_lengths(input_lengths) # (B)
        
            # Расчет Loss
            valid_indices = output_lengths >= target_lengths
            valid_indices &= (output_lengths > 0) & (target_lengths > 0)
            
            if valid_indices.any():
                outputs_f = outputs[valid_indices]
                targets_list = []
                current_pos = 0
                valid_tgt_lens = []
                target_lengths_cpu = target_lengths.cpu()
                targets_cpu = targets.cpu()
                
                for i in range(len(target_lengths_cpu)): 
                    ln = target_lengths_cpu[i].item()
                    if valid_indices[i]: 
                        targets_list.append(targets_cpu[current_pos : current_pos + ln])
                        valid_tgt_lens.append(ln)
                        current_pos += ln
                if targets_list:
                    targets_f = torch.cat(targets_list).to(device)
                    out_lens_f = output_lengths[valid_indices]
                    tgt_lens_f = torch.tensor(valid_tgt_lens, dtype=torch.long, device=device)
                    
                    if targets_f.numel() > 0:
                        log_probs_f = outputs_f.permute(1, 0, 2)
                        loss = ctc_loss(log_probs_f, targets_f, out_lens_f, tgt_lens_f)
                        
                        if not (torch.isnan(loss) or torch.isinf(loss)):
                            val_loss_accum_gpu += loss.item() * outputs_f.size(0) # Суммируем локально
                        processed_samples_val_gpu += outputs_f.size(0) # Считаем локально
        except Exception as e:
            print(f"Ошибка model forward/get_output_lengths в evaluate_wer: {e}")
            # Попытка вывести размеры для отладки
            print(f"Форма inputs: {inputs.shape}, Длины входа T: {input_lengths}")
            continue

        preds_indices = torch.argmax(outputs, dim=2) # (B, T_out)
        preds_indices_cpu = preds_indices.cpu().tolist()
        output_lengths_cpu = output_lengths.cpu().tolist()

        for i in range(len(inputs)):
            actual_len = output_lengths_cpu[i]
            pred_text = ""
            
            if actual_len > 0 :
                pred_text = int_to_text(preds_indices_cpu[i][:actual_len], index_map)
            
            local_predictions.append(pred_text)
        local_references.append(ref_texts)

        # Собираем лосс
        total_val_loss_tensor = torch.tensor([val_loss_accum_gpu], device=device)
        total_val_samples_tensor = torch.tensor([processed_samples_val_gpu], device=device)
        dist.all_reduce(total_val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_val_samples_tensor, op=dist.ReduceOp.SUM)

        # Рассчитываем средний лосс
        avg_val_loss = (total_val_loss_tensor[0] / total_val_samples_tensor[0]).item() if total_val_samples_tensor[0] > 0 else float('inf')

        # Собираем списки предсказаний и выводов на мастере (rank 0)
        gathered_preds_list = [None] * world_size
        gathered_refs_list = [None] * world_size

        # Каждый процесс передает свой локальный список CPU
        dist.gather_object(local_predictions, gathered_preds_list if is_master else None, dst=0)
        dist.gather_object(local_references, gathered_refs_list if is_master else None, dst=0)

    calculated_wer = float('inf')
    if is_master:
        # Объединяем списки со всех процессов
        flat_preds = [item for sublist in gathered_preds_list for item in sublist]
        flat_refs = [item for sublist in gathered_refs_list for item in sublist]

        val_preds = flat_preds # Сохраняем для вывода примеров
        val_refs = flat_refs

        num_samples_val_total = len(dataloader) # Общий размер валидационного датасета

        # Обрезаем до нужного размера (на случай drop_last=False)
        flat_preds = flat_preds[:num_samples_val_total]
        flat_refs = flat_refs[:num_samples_val_total]

        print("Расчет WER...")
        if flat_refs and flat_preds and len(flat_refs) == len(flat_preds):
            try:
                epoch_wer = jiwer.wer(flat_refs, flat_preds) * 100
                print(f"Ошибка WER = {epoch_wer:.2f}%")
            except Exception as e:
                print(f"Ошибка при расчете WER: {e}")
        else:
            print(f"Не совпали длины для WER ({len(flat_refs)}/{len(flat_preds)}).")

        # Синхронизация WER для всех процессов
        wer_tensor = torch.tensor([epoch_wer], device=device)   # Мастер имеет значение, остальные inf
        dist.broadcast(wer_tensor, src=0)                       # Рассылаем значение с мастера
        epoch_wer = wer_tensor[0].item()

        if val_preds and val_refs:
            print("  Примеры предсказаний (Pred | Target):")
            for i in range(min(5, len(val_preds))):
                print(f"    - '{val_preds[i]}' | '{val_refs[i]}'")

    return epoch_wer, avg_val_loss 


# Функция для настройки параллельных вычислений
def setup(rank, world_size, master_addr, master_port, backend='nccl'):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    # Инициализация группы процессов
    # rank - уникальный номер текущего процесса (от 0 до world_size-1)
    # world_size - общее количество процессов (машин * GPU на машину)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    torch.cuda.set_device(0) # Используем GPU с индексом 0 на этой машине
    print(f"Rank {rank}. Распределенная среда инициализирована на {torch.cuda.current_device()}.")

# Функция для очистки 
def cleanup():
    dist.destroy_process_group()
    print("Группа процессов уничтожена.")

if __name__ == "__main__":
    # При запуске через torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", -1))        # Локальный ранг на машине
    world_size = int(os.environ.get("WORLD_SIZE", -1))        # Общее число процессов
    rank = int(os.environ.get("RANK", -1))                    # Глобальный ранг
    master_addr = os.environ.get("MASTER_ADDR", -1)           # Адрес мастер-узла
    master_port = os.environ.get("MASTER_PORT", -1)           # Порт мастер-узла

    print(f"Запуск с: Rank={rank}, World Size={world_size}, Local Rank={local_rank}, Master={master_addr}:{master_port}")

    # Гиперпараметры
    INPUT_DIM = 80                                          # n_mels число уровней мэл спектрограммы
    HIDDEN_DIM = 1024                                       # Число скрытых признаков LSTM
    OUTPUT_DIM = len(RUSSIAN_ALPHABET)                      # Алфавит
    NUM_LAYERS = 7                                          # Число слоев LSTM
    DROPOUT = 0.3                                           # Обрубание весов
    BATCH_SIZE = 64                                         # Число обрабатываемых аудио за проход
    NUM_EPOCHS = 50                                         # Число эпох обучения
    LEARNING_RATE = 4e-4                                    # Adam с большими моделями лучше сходится с меньшим LR
    WEIGHT_DECAY = 1e-4                                     # Небольшая L2 регуляризация
    CLIP_GRAD_NORM = 5.0                                    # Для предотвращения взрыва градиентов
    MODEL_SAVE_PATH = ""                                    # Путь сохранения модели
    MODEL_PATH = ""                                         # Путь для дообучения модели
    TRAIN_SPLIT_RATIO = 0.9                                 # 90% на обучение, 10% на валидацию
    PATIENCE_SCHEDULER = 3                                  # для ReduceLROnPlateau
    PATIENCE_EARLY_STOPPING = 15                            # Число эпох без улучшения для выхода
    FREQ_MASK_PARAM = 27                                    # Аугментация по частоте
    TIME_MASK_PARAM = 70                                    # Аугментация по времени

    # Создание папок (только ранк 0)
    if rank == 0:
        if not os.path.exists("./runs/"): 
            os.makedirs("./runs/")

        if not os.path.exists("./models/"): 
            os.makedirs("./models/")

    print(f"[Rank {rank}] Начало функции обучения.")
    setup(rank, world_size, master_addr, master_port)

    device = torch.device(f"cuda:{0}")                       # Устройство для этого процесса
    is_master = (rank == 0)                                  # Определяем мастер-процесс

    data_sources = [
        ("/kaggle/input/audiosets/dataset_target.csv", "/kaggle/input/audiosets/asr_public_phone_calls_1/asr_public_phone_calls_1/0"),
        ("/kaggle/input/russian-asr-golos/golos/golos/dataset_target.csv", "/kaggle/input/russian-asr-golos/golos/0")
        ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    # Инициализация TensorBoard
    writer = None
    if is_master:
        run_name = f"asr_layer{NUM_LAYERS}_neurons{HIDDEN_DIM}_lr{LEARNING_RATE}_wd{WEIGHT_DECAY}_dr{DROPOUT}"
        log_dir = os.path.join("runs", run_name)
        writer = SummaryWriter(log_dir)
        print(f"Логи TensorBoard: {log_dir}")

    # Загрузка и разделение данных
    if is_master: 
        print("Создание датасетов...")

    all_datasets_train_version = [
            SingleSourceAudioDataset(csv, adir, char_map, apply_augmentation=False)
            for csv, adir in data_sources
        ]

    all_datasets_val_version = [
            SingleSourceAudioDataset(csv, adir, char_map, apply_augmentation=False)
            for csv, adir in data_sources
        ]

    # Фильтруем пустые Datasets
    all_datasets_train_version = [ds for ds in all_datasets_train_version if len(ds) > 0]
    all_datasets_val_version = [ds for ds in all_datasets_val_version if len(ds) > 0]
    
    if not all_datasets_train_version:
        print("Ошибка: Нет валидных данных!")
        exit()

    combined_dataset_train_ver = ConcatDataset(all_datasets_train_version)
    combined_dataset_val_ver = ConcatDataset(all_datasets_val_version)
    dataset_size = len(combined_dataset_train_ver)
    
    if is_master:
        print(f"Общий размер объединенного датасета: {dataset_size}")

    train_size = int(TRAIN_SPLIT_RATIO * dataset_size)
    val_size = dataset_size - train_size
    train_indices, val_indices = random_split(range(dataset_size), [train_size, val_size])

    train_subset = Subset(combined_dataset_train_ver, train_indices)    # Train БЕЗ аугментации
    val_subset = Subset(combined_dataset_val_ver, val_indices)          # Val БЕЗ аугментации

    if is_master:
        print(f"Размер обучающей выборки: {len(train_subset)}")
        print(f"Размер валидационной выборки: {len(val_subset)}")

    # Для параллельных вычислений для каждой машины часть датасета
    train_sampler = DistributedSampler(train_subset, 
                                       num_replicas=world_size,
                                       rank=rank,
                                       shuffle=True,
                                       drop_last=True
                                      )
    val_sampler = DistributedSampler(val_subset,
                                     num_replicas=world_size,
                                     rank=rank,
                                     shuffle=False
                                    )

    # DataLoaders
    train_dataloader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn_asr_wer, num_workers=4, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_subset, batch_size=BATCH_SIZE, sampler=val_sampler, collate_fn=collate_fn_asr_wer, num_workers=4, pin_memory=True, drop_last=False)

    # Инициализация модели
    model = ASR_CTC_Model(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT).to(device)

    # Загрузка весов (если нужно, выполняется до DDP)
    if os.path.exists(MODEL_PATH):
        if is_master:
            print(f"Загрузка весов из {MODEL_PATH}")
        
        # Загружаем на CPU, чтобы избежать проблем с device_map
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        model.load_state_dict(state_dict)
        model.to(device)
        if is_master: 
            print("Веса загружены.")
    else:
        if is_master: 
            print(f"Предупреждение: Файл {MODEL_PATH} не найден.")

    model = DDP(model, device_ids=[local_rank], output_device=local_rank) # Загрузка на GPU 0

    if is_master:
        print(model)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Общее количество обучаемых параметров: {total_params:,}")

    # Loss, Optimizer, Scheduler
    ctc_loss = nn.CTCLoss(blank=char_map[BLANK_CHAR], reduction='mean', zero_infinity=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE_SCHEDULER) # mode='min' для WER

    # Цикл обучения
    best_val_wer = float('inf')
    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 0 # Для продолжения обучения

    if is_master:
        print(f"\n Начало обучения ({NUM_EPOCHS} эпох) ")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        start_time_epoch = time.time()

        # Обучение начало
        model.train()
        train_sampler.set_epoch(epoch) # Для всех машин перемешивание сэмплов
        
        train_loss_accum = 0.0
        processed_samples_train = 0

        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Training", disable=not is_master)

        # ОБУЧЕНИЕ основной цикл
        for batch_idx, batch_data in enumerate(train_pbar):
            if batch_data is None or batch_data[0] is None:
                continue

            try:
                inputs, targets_concat, input_lengths_T, target_lengths, _ = batch_data
            except ValueError as e:
                print(f"\nОшибка распаковки трен. батча {batch_idx+1}: {e}")
                continue

            # Проверка на пустые тензоры после фильтрации в collate
            if inputs.numel() == 0 or targets_concat.numel() == 0:
                 continue

            inputs = inputs.to(device, non_blocking=True)
            targets = targets_concat.to(device, non_blocking=True)
            input_lengths = input_lengths_T.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            try:
                outputs = model(inputs) # (B, T_out, C)
                output_lengths = model.module.get_output_lengths(input_lengths).to(device) # (B)

                # Проверка валидности длин полученной и целевой
                valid_indices = output_lengths >= target_lengths
                valid_indices &= (output_lengths > 0) & (target_lengths > 0)

                if not valid_indices.any(): # Если нет валидных в батче
                    # print(f"\nПропущен батч {batch_idx+1}: нет валидных длин.")
                    continue

                # Фильтруем батч для лосса, если не все валидны
                if not valid_indices.all():
                    outputs = outputs[valid_indices]            # Выбор только валидных элементов
                    targets_list = []
                    current_pos = 0
                    valid_target_lengths_list = []
                    target_lengths_cpu = target_lengths.cpu()   # Для итерации
                    targets_cpu = targets.cpu()                 # Для срезов

                    # Создание нового конкатенированного тензора для outputs
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
            if is_master and writer and (batch_idx + 1) % 1000 == 0:
                global_step = epoch * len(train_dataloader) + batch_idx
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)

        # Собираем метрики с машины
        total_loss_tensor = torch.tensor([train_loss_accum], device=device)             # Ошибка на текущей машине
        total_samples_tensor = torch.tensor([processed_samples_train], device=device)   # Число обработанных сэмплов
        
        # C помощью dist.all_reduce все машины узнают общее число
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

        avg_train_loss = (total_loss_tensor[0] / total_samples_tensor[0]).item() if total_samples_tensor[0] > 0 else float('inf')

        # Валидация 
        # Переключаем модель в eval режим ВНУТРИ evaluate_wer
        epoch_wer, avg_val_loss = evaluate_wer(model, val_dataloader, device, index_map)

        epoch_duration = time.time() - start_time_epoch

        if is_master:
            # Логирование эпохи
            writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch + 1) # Логируем по номеру эпохи (начиная с 1)
            writer.add_scalar('Loss/validation_epoch', avg_val_loss, epoch + 1)
            writer.add_scalar('WER/validation', epoch_wer, epoch + 1)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch + 1)
            writer.add_scalar('EpochDuration_sec', epoch_duration, epoch + 1)

            # Вывод результатов эпохи
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} Summary:")
            print(f"  Duration: {epoch_duration:.2f}s")
            print(f"  Avg Train Loss: {avg_train_loss:.4f}")
            print(f"  Avg Val Loss: {avg_val_loss:.4f}")
            print(f"  Validation WER: {epoch_wer:.2f}%")
            print(f"  Current LR: {optimizer.param_groups[0]['lr']:.3e}")
            print("  Example Predictions (Predicted | Target):")

        # Обновление планировщика, сохранение лучшей модели, ранняя остановка
        scheduler.step(epoch_wer)

        if is_master:
            if epoch_wer < best_val_wer:
                cur_epoch = epoch
                best_val_wer = epoch_wer
                MODEL_SAVE_PATH = f"./models/asr_improved_layers{NUM_LAYERS}_hd{HIDDEN_DIM}_epoch{cur_epoch}_WER{int(epoch_wer)}.pth"
                torch.save(model.module.state_dict(), MODEL_SAVE_PATH)
                print(f"  Validation WER улучшился до {best_val_wer:.2f}%. Модель сохранена.")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                torch.save(model.module.state_dict(), f"./models/asr_unchanged_layers{NUM_LAYERS}_hd{HIDDEN_DIM}_epoch{cur_epoch}_WER{int(epoch_wer)}.pth")
                print(f"  Validation WER не улучшился. Эпох без улучшения: {epochs_no_improve}/{PATIENCE_EARLY_STOPPING}")

        # Синхронизация Early Stopping
        stop_flag = torch.tensor([1.0 if epochs_no_improve >= PATIENCE_EARLY_STOPPING else 0.0], device=device)
        dist.broadcast(stop_flag, src=0) # Мастер рассылает остальным
        should_stop = stop_flag.item() > 0.5

        if should_stop:
            if is_master: 
                print(f"\nРанняя остановка на эпохе {epoch+1}!")
            break # Все процессы выходят

        if is_master: 
            print("-" * 50)

        try:
            gc.collect()
        except:
            if is_master:
                print("Сборщик мусора вызвал исключение\n")

        # Барьер для синхронизации перед следующей эпохой
        dist.barrier()

    if writer and is_master: 
        writer.close()
    if is_master:
        print(f"Обучение завершено. Лучший Validation WER: {best_val_wer:.2f}% сохранен в {MODEL_SAVE_PATH}")

    cleanup() # Завергение работы машин
