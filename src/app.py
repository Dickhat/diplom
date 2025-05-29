import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sounddevice as sd
import numpy as np
import threading
import os
import datetime
import queue
import time

# Импорты для модели и обработки
import torch
import torch.nn as nn
import librosa
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration

# КОНСТАНТЫ И ФУНКЦИИ МОДЕЛИ 
RUSSIAN_ALPHABET = "_абвгдеёжзийклмнопрстуфхцчшщъыьэюя "
BLANK_CHAR = '_'
char_map = {char: idx for idx, char in enumerate(RUSSIAN_ALPHABET)}
index_map = {idx: char for char, idx in char_map.items()}

def text_to_int(text, char_map):
    text = text.lower()
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
        # Проверяем наличие индекса в карте перед добавлением
        if idx in index_map:
            text += index_map[idx]
        last_idx = idx
    text = ' '.join(text.split())
    return text

# Функция предобработки для получения лог-мел-спетрограммы
def preprocess_buffer(audio_chunk: np.ndarray, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160):
    """Преобразует NumPy массив аудио в лог-мел-спектрограмму (F, T)."""

    # Убедимся, что данные в формате float32, как ожидает librosa
    if audio_chunk.dtype != np.float32:
        audio_chunk = audio_chunk.astype(np.float32)

    # Дополнение тишиной на всякий случай (хотя из буфера должна приходить нужная длина)
    if len(audio_chunk) < n_fft:
        padding_needed = n_fft - len(audio_chunk)
        audio_chunk = np.pad(audio_chunk, (0, padding_needed), mode='constant', constant_values=0)

    # import matplotlib.pyplot as plt


    # # Построим график амплитудных отсчётов
    # plt.figure(figsize=(10, 4))
    # plt.plot(audio_chunk)
    # plt.title("Амплитудные отсчёты аудиосигнала")
    # plt.xlabel("Номер отсчёта")
    # plt.ylabel("Амплитуда")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    try:
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_chunk, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=8000)
    except Exception as e:
         print(f"Ошибка при вычислении melspectrogram из буфера: {e}")
         return None
    
    if np.max(mel_spectrogram) < 1e-10: return None # Пропускаем тишину

    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    mean = np.mean(log_mel_spectrogram)
    std = np.std(log_mel_spectrogram)
    
    if std < 1e-6: 
        return None
    
    log_mel_spectrogram = (log_mel_spectrogram - mean) / (std + 1e-6)
    
    return torch.tensor(log_mel_spectrogram, dtype=torch.float32) # (F, T)

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

# ПАРАМЕТРЫ МОДЕЛИ И ПУТЬ
INPUT_DIM = 80
HIDDEN_DIM = 1024 # Число нейронов в скрытом слое LSTM
OUTPUT_DIM = len(RUSSIAN_ALPHABET)
NUM_LAYERS = 3    # Число слоев LSTM
DROPOUT = 0.3   
#MODEL_LOAD_PATH = "D:/models/1024_3layer_all_datasets/all_data_1024_3layer_epoch9_WER29.26.pth"
MODEL_LOAD_PATH = "D:/models/ASR_CTC_MODEL_1024_3layer_GOLOS, common voice, sova ai, librispeech/gslc_asr_ctc_model_1024_3layer_epoch20_WER11.60.pth" # Пример

# ОПРЕДЕЛЕНИЕ УСТРОЙСТВА
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство для инференса: {device}")

# ЗАГРУЗКА МОДЕЛИ 
print("Загрузка модели...")
model = ASR_CTC_Model(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT).to(device)
try:
    # map_location=device гарантирует загрузку на правильное устройство
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
    model.eval() # <--- ОБЯЗАТЕЛЬНО переводим модель в режим оценки
    print("Модель успешно загружена и переведена в режим оценки.")
except FileNotFoundError:
    print(f"Ошибка: Файл модели не найден по пути: {MODEL_LOAD_PATH}")
    print("Транскрибация будет невозможна.")
    model = None # Ставим модель в None, чтобы не было ошибок дальше
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    model = None

# Функция транскрибации чанка
@torch.no_grad() # Отключаем градиенты для инференса
def transcribe_chunk(audio_chunk, model, device, index_map):
    """Транскрибирует один чанк аудио (NumPy array)."""
    if model is None: # Если модель не загрузилась
        return "[Модель не загружена]"

    # Предобработка буфера -> Тензор (F, T) на CPU
    input_features = preprocess_buffer(audio_chunk) # sample_rate и др. параметры берутся по умолчанию

    if input_features is None:
        # print("Предобработка не удалась.")
        return "[Ошибка обработки]"

    # Перенос на device и добавление batch-измерения -> (1, F, T)
    input_tensor = input_features.unsqueeze(0).to(device)
    # Длина по времени (T) для get_output_lengths
    input_length_T = torch.tensor([input_features.shape[1]], dtype=torch.long).to(device)

    try:
        # Получение выхода модели
        outputs = model(input_tensor) # (1, T_out, C)
        # Получение длины выхода
        output_lengths = model.get_output_lengths(input_length_T) # (1)

        if output_lengths[0] <= 0:
            # print("Нулевая длина выхода модели.")
            return "" # Пустая строка, если выход нулевой

        
        # Greedy декодирование
        pred_indices = torch.argmax(outputs, dim=2).squeeze(0) # (T_out)
        pred_indices_cpu = pred_indices[:output_lengths[0]].cpu().tolist() # Обрезаем и переносим на CPU
        decoded_text = int_to_text(pred_indices_cpu, index_map)

        return decoded_text

    except Exception as e:
        print(f"Ошибка во время инференса модели: {e}")
        return "[Ошибка инференса]"

QUEUE_END_MARKER = object()

# Класс GUI приложения
class AudioApp:
    def __init__(self, root, loaded_model, feauture_extractor=None, whisper_model=None, tokenizer=None):
        self.root = root
        self.root.title("Автоматическая транскрибация речи")
        self.model = loaded_model # Модель для моей
        self.whisper_feature_extractor = feauture_extractor
        self.whisper_model = whisper_model
        self.whisper_tokenizer = tokenizer
        self.device = device      # CPU или GPU

        #  Параметры записи
        self.samplerate = 16000
        self.channels = 1
        self.blocksize = 1024 # Размер блока от sounddevice
        self.seconds_per_chunk = 5 # Сколько секунд накапливать для транскрибации
        self.samples_per_chunk = self.samplerate * self.seconds_per_chunk

        # GUI элементы
        self.save_directory = os.getcwd()

        top_frame = ttk.Frame(root)                 # Создание группирующего виджета
        top_frame.pack(fill=tk.X, padx=10, pady=5)  # Соединение top_frame с root отступы 10 и 5 и растягивание по горизонтали

        # Создание надписи по правому краю top_frame
        self.dir_label = ttk.Label(top_frame, text="📁 Папка: " + self.save_directory, anchor="e")
        self.dir_label.pack(side=tk.RIGHT)

        # Создание кнопки в правом краю
        self.dir_button = ttk.Button(top_frame, text="Выбрать папку", command=self.choose_directory)
        self.dir_button.pack(side=tk.RIGHT, padx=5)

        # Создание кнопки с отступом вниз на 10
        self.record_button = ttk.Button(root, text="Начать запись", command=self.toggle_recording)
        self.record_button.pack(pady=10)

        # Создание индикатора под кнопкой записи
        self.recording_indicator = ttk.Label(root, text="", foreground="red", font=("Arial", 10, "bold"))
        self.recording_indicator.pack()
        
        # Область для вывода текста
        self.transcription_text = tk.Text(root, height=10, width=60, wrap=tk.WORD, state=tk.DISABLED, font=("Arial", 10))
        self.transcription_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Добавим прокрутку текста
        scrollbar = ttk.Scrollbar(self.transcription_text, command=self.transcription_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.transcription_text['yscrollcommand'] = scrollbar.set

        try:
            default_input = sd.default.device[0]  # индекс устройства ввода
            if default_input is None or default_input < 0:
                messagebox.showwarning("Внимание", "Микрофон не найден!")
            else:
                messagebox.showinfo("Уведомление", "Микрофон подключен!")
        except:
            messagebox.showwarning("Внимание", "Микрофон не найден!")

        # Очередь и буфер
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.internal_buffer = [] # Накапливаем здесь до нужного размера

        # Состояние
        self.recording_thread = None
        self.processing_thread = None
        self.is_recording = False                   # Флаг для основного потока записи
        self.gui_update_timer = None                # Для таймера обновления GUI
        self.gui_update_ms = 100                    # Как часто проверять очередь текста (мс)

    def choose_directory(self):
        folder = filedialog.askdirectory()
        if folder: 
            self.save_directory = folder
            self.dir_label.config(text="📁 Папка: " + self.save_directory)

    def audio_callback(self, indata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
        """Callback функция от sounddevice."""
        if status: print(status, flush=True) # Выводим статус, если есть
        if not self.is_recording: return

        # Копируем данные и добавляем в буфер и берем только первый канал
        self.internal_buffer.append(indata[:, 0].copy())

        # Проверяем, накопилось ли достаточно сэмплов (shape[0] размерность)
        total_samples_in_buffer = sum(chunk.shape[0] for chunk in self.internal_buffer)

        if total_samples_in_buffer >= self.samples_per_chunk:
            # Объединяем части буфера в один длинный сегмент
            concatenated_audio = np.concatenate(self.internal_buffer, axis=0)

            # Берем ровно samples_per_chunk
            chunk_to_process = concatenated_audio[:self.samples_per_chunk]
            
            # Оставшиеся данные возвращаем в буфер
            remaining_audio = concatenated_audio[self.samples_per_chunk:]

            # Хвост аудиоданных для обработки с "памятью"
            #tail_audio = chunk_to_process[-1600:]

            # Кладем готовый чанк в очередь
            self.audio_queue.put(chunk_to_process)

            # Обновляем внутренний буфер
            if remaining_audio.size > 0:
                self.internal_buffer = [remaining_audio] #[tail_audio, remaining_audio]
            else:
                self.internal_buffer = [] #[tail_audio]

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else: 
            self.stop_recording()

    def start_recording(self):
        if self.is_recording:
            return # Уже запущено
        
        print("Запуск записи...")

        self.is_recording = True
        self.internal_buffer = [] # Очищаем буфер
        self.record_button.config(text="Остановить запись")
        self.recording_indicator.config(text="● Идёт запись...")
        self.clear_transcription() # Очищаем текстовое поле

        # Запускаем поток для обработки аудио из очереди
        self.start_processing_thread()
        self.start_gui_updater() # Запускаем обновление GUI

        # Функция для выполнения в потоке записи
        def _record_thread():
            try:
                # контекстный менеджер для потока
                with sd.InputStream(callback=self.audio_callback,
                                    samplerate=self.samplerate,
                                    channels=self.channels,
                                    blocksize=self.blocksize, # Размер блока для callback
                                    dtype='float32'): # Используем float32
                    while self.is_recording:
                        sd.sleep(100) # Небольшая пауза, чтобы не грузить CPU
                print("Поток записи завершен.")
            except Exception as e:
                messagebox.showerror("Ошибка", "Запись не может быть начата без записывающего устройства")

                print(f"Ошибка в потоке записи: {e}")
                self.root.after(0, self.stop_recording) # Вызываем stop_recording в главном потоке

        # Запускаем поток записи
        self.recording_thread = threading.Thread(target=_record_thread, daemon=True)
        self.recording_thread.start()

    def stop_recording(self):
        if not self.is_recording: return # Уже остановлено
        
        print("Остановка записи...")
        self.is_recording = False # Сигнализируем потоку записи остановиться

        # Ждем завершения потока записи (с таймаутом)
        if self.recording_thread is not None:
            print("Ожидание завершения потока записи...")
            self.recording_thread.join(timeout=1.0) # Ждем не более 1 секунд
            if self.recording_thread.is_alive():
                print("Предупреждение: Поток записи не завершился вовремя.")
            self.recording_thread = None

        # Проверка, что буфер пуст
        if self.internal_buffer:
            # Объединяем все, что осталось
            final_chunk = np.concatenate(self.internal_buffer, axis=0)
            if final_chunk.size > 0: # Только если что-то есть
                self.audio_queue.put(final_chunk) # Кладем последний кусок в очередь

            self.internal_buffer = [] # Очищаем буфер

        self.audio_queue.put(QUEUE_END_MARKER)

        # Ждем завершения потока обработки (он должен доопработать очередь)
        if self.processing_thread is not None:
            print("Ожидание завершения потока обработки...")
            self.audio_queue.join() # Даем БОЛЬШЕ времени на дообработку
            self.processing_thread.join()
            self.processing_thread = None
        print("Поток обработки остановлен.")

        self.stop_gui_updater()

        self.check_transcription_queue(run_once=True)
        self.root.update_idletasks()

        self.save_transcription()

        self.record_button.config(text="Начать запись")
        self.recording_indicator.config(text="")
        print("Запись остановлена.")

    def save_transcription(self):
        """Сохраняет весь текст из текстового поля в файл."""
        full_text = self.transcription_text.get("1.0", tk.END).strip() # Получаем весь текст
        
        if not full_text:
            print("Нет текста для сохранения.")
            self.dir_label.config(text="Нет текста для сохранения.")
            return

        filename = datetime.datetime.now().strftime("transcription_%d_%m_%Y_%H-%M-%S.txt")

        path = os.path.join(self.save_directory, filename)

        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            print(f"Транскрипция сохранена: {path}")

            # basename для короткого сообщения в UI
            self.dir_label.config(text=f"✅ Текст сохранен: {os.path.basename(path)}")
        except Exception as e:
            print(f"Ошибка сохранения транскрипции: {e}")
            self.dir_label.config(text="⚠️ Ошибка сохранения текста!")

    def start_processing_thread(self):
        """Запускает поток обработки очереди."""
        self.processing_thread = threading.Thread(target=self.process_audio_from_queue, daemon=True)
        self.processing_thread.start()

    def process_audio_from_queue(self):
        """Поток, который берет чанки из очереди и транскрибирует их."""
        print("Поток обработки очереди запущен.")

        if self.whisper_model:
            forced_decoder_ids = self.whisper_tokenizer.get_decoder_prompt_ids(language="ru", task="transcribe")

        transcribed_text = ""
        while True:
            try:
                # Ждем данные из очереди (с таймаутом, чтобы можно было подождать данные)
                audio_chunk = self.audio_queue.get(timeout=0.5) # Таймаут 0.5 сек

                if audio_chunk is QUEUE_END_MARKER:
                    print("Поток обработки: получен маркер конца.")
                    self.audio_queue.task_done()
                    break

                rms = np.sqrt(np.mean(audio_chunk**2))  # Корень из среднего квадрата — RMS
                db = 20 * np.log10(rms + 1e-9)          # Преобразуем в dB, добавляем ε чтобы не делить на 0

                if db < -60:  
                    self.audio_queue.task_done()
                    #print("Тишина, пропускаем...")
                    #print(f"db {db}")
                    continue
                
                #print(f"Не шум db {db}")

                if self.whisper_model:
                    feature = self.whisper_feature_extractor(audio_chunk, sampling_rate=16000, language="ru", return_tensors="pt").input_features
                    feature = feature.to(self.device)

                    generated_tokens =self.whisper_model.generate(input_features=feature, forced_decoder_ids=forced_decoder_ids)
                    
                    transcribed_text = self.whisper_tokenizer.batch_decode(generated_tokens.cpu(), skip_special_tokens=True)[0]
                else:
                    # Транскрибация моей моделью
                    transcribed_text = transcribe_chunk(audio_chunk, self.model, self.device, index_map)

                # Обновление GUI
                if transcribed_text and transcribed_text not in ["[Ошибка обработки]", "[Ошибка инференса]", "[Модель не загружена]"]:
                    self.transcription_queue.put(transcribed_text)

                self.audio_queue.task_done()
            except Exception as e:
                #print(f" Очередь пустая: {e}")
                time.sleep(0.1) # Небольшая пауза при ошибке    
        
        try: 
            self.audio_queue.task_done() # Пытаемся пометить задачу как выполненную
        except ValueError: 
            pass
        print("Поток обработки очереди остановлен.")

    # обновления GUI из очереди
    def check_transcription_queue(self, run_once=False):
        """Проверяет очередь текста и обновляет виджет."""
        try:
            while True: # Обрабатываем все сообщения в очереди
                text_to_add = self.transcription_queue.get_nowait()
                self.update_transcription_widget(text_to_add + " ")
        except queue.Empty:
            pass # Очередь пуста
        except Exception as e:
            print(f"Ошибка обновления GUI из очереди: {e}")
        finally:
            # Планируем следующую проверку, только если не запрошен однократный запуск
            # и если окно еще существует
            if not run_once and self.root.winfo_exists():
                self.gui_update_timer = self.root.after(self.gui_update_ms, self.check_transcription_queue)

    def update_transcription_widget(self, text_to_add):
        """Безопасно обновляет текстовое поле из главного потока."""
        self.transcription_text.config(state=tk.NORMAL) # Разрешаем редактирование
        self.transcription_text.insert(tk.END, text_to_add)
        self.transcription_text.see(tk.END) # Автопрокрутка вниз
        self.transcription_text.config(state=tk.DISABLED) # Снова запрещаем

    def clear_transcription(self):
        """Очищает текстовое поле."""
        self.transcription_text.config(state=tk.NORMAL)
        self.transcription_text.delete('1.0', tk.END)
        self.transcription_text.config(state=tk.DISABLED)

    # управление таймером обновления GUI 
    def start_gui_updater(self):
        """Запускает периодическую проверку очереди текста."""
        self.stop_gui_updater()             # Останавливаем предыдущий таймер, если был
        self.check_transcription_queue()    # Запускаем первый раз

    def stop_gui_updater(self):
        """Останавливает периодическую проверку очереди текста."""
        if self.gui_update_timer is not None:
            self.root.after_cancel(self.gui_update_timer)
            self.gui_update_timer = None

    def on_closing(self):
        """Обработка закрытия окна."""
        print("Закрытие приложения...")
        if self.is_recording:
            self.stop_recording() # Останавливаем запись, если идет
        else:
            if self.processing_thread is not None and self.processing_thread.is_alive():
                print("Отправка маркера конца при закрытии...")
                self.audio_queue.put(QUEUE_END_MARKER)
                # Не ждем здесь долго, просто даем шанс завершиться
                self.processing_thread.join(timeout=1.0)
        
        self.stop_gui_updater() # Останавливаем таймер GUI
        self.root.destroy()     # Закрытие окна

if __name__ == "__main__":
    whisper_model = None
    feauture_extractor = None
    tokenizer = None

    # whisper-tiny
    # feauture_extractor = WhisperFeatureExtractor.from_pretrained("models--openai--whisper-tiny/snapshots/169d4a4341b33bc18d8881c4b69c2e104e1cc0af")
    # whisper_model = WhisperForConditionalGeneration.from_pretrained("models--openai--whisper-tiny/snapshots/169d4a4341b33bc18d8881c4b69c2e104e1cc0af")
    # tokenizer = WhisperTokenizer.from_pretrained("models--openai--whisper-tiny/snapshots/169d4a4341b33bc18d8881c4b69c2e104e1cc0af", language = "russian", task="transcribe")

    # whisper-large-v3-turbo
    # feauture_extractor = WhisperFeatureExtractor.from_pretrained("models--openai--whisper-large-v3-turbo/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9")
    # whisper_model = WhisperForConditionalGeneration.from_pretrained("models--openai--whisper-large-v3-turbo/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9")
    # tokenizer = WhisperTokenizer.from_pretrained("models--openai--whisper-large-v3-turbo/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9", language = "russian", task="transcribe")
    
    # whisper_model.to(device)

    root = tk.Tk()

    if whisper_model:
        app = AudioApp(root, model, feauture_extractor, whisper_model, tokenizer) # Загруженная модель Whisper
    else:
        app = AudioApp(root, model) # Загруженная модель моя
    
    root.protocol("WM_DELETE_WINDOW", app.on_closing) # Обработка закрытия окна
    root.mainloop()