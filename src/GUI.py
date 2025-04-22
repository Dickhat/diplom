import tkinter as tk
from tkinter import ttk, filedialog
import sounddevice as sd
import numpy as np
import threading
import os
from scipy.io.wavfile import write as write_wav
import datetime
import queue

samplerate = 16000
channels = 1
blocksize = 1024
SECONDS = 1
samples_per_chunk = samplerate * SECONDS

class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Запись голоса в реальном времени")

        self.save_directory = os.getcwd()

        # Верхняя панель с кнопкой выбора папки
        top_frame = ttk.Frame(root)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        self.dir_label = ttk.Label(top_frame, text="📁 Папка: " + self.save_directory, anchor="e")
        self.dir_label.pack(side=tk.RIGHT)

        self.dir_button = ttk.Button(top_frame, text="Выбрать папку", command=self.choose_directory)
        self.dir_button.pack(side=tk.RIGHT, padx=5)

        # Кнопка записи
        self.record_button = ttk.Button(root, text="Начать запись", command=self.toggle_recording)
        self.record_button.pack(pady=10)

        # Индикатор записи
        self.recording_indicator = ttk.Label(root, text="", foreground="red", font=("Arial", 10, "bold"))
        self.recording_indicator.pack()

        # Подпись
        self.transcription_label = ttk.Label(root, text="Здесь появляется транскрибированный текст", foreground="gray", font=("Arial", 10, "italic"))
        self.transcription_label.pack(pady=10)

        # Очередь для передачи данных
        self.audio_queue = queue.Queue()
        self.audio_buffer = []

        # Флаг записи
        self.running = False
        self.latest_data = np.zeros(blocksize)
        self.recorded_data = []

        # Запускаем поток для обработки аудио
        threading.Thread(target=self.process_audio_from_queue, daemon=True).start()

    def choose_directory(self):
        folder = filedialog.askdirectory()
        if folder:
            self.save_directory = folder
            self.dir_label.config(text="📁 Папка: " + self.save_directory)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)

        self.latest_data = indata[:, 0]  # для отображения
        self.audio_buffer.append(indata.copy())

        # Проверяем, накопилось ли 1 секунда
        total_samples = sum(chunk.shape[0] for chunk in self.audio_buffer)
        if total_samples >= samples_per_chunk:
            # Объединяем и кладём в очередь
            chunk = np.concatenate(self.audio_buffer, axis=0)[:samples_per_chunk]
            self.audio_queue.put(chunk)
            # Удаляем использованные данные из буфера
            remaining = np.concatenate(self.audio_buffer, axis=0)[samples_per_chunk:]
            self.audio_buffer = [remaining] if len(remaining) > 0 else []

    def toggle_recording(self):
        if not self.running:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.running = True
        self.recorded_data = []
        self.record_button.config(text="Остановить запись")
        self.recording_indicator.config(text="● Идёт запись...")

        def _thread():
            with sd.InputStream(callback=self.audio_callback,
                                channels=channels,
                                samplerate=samplerate,
                                blocksize=blocksize):
                while self.running:
                    sd.sleep(100)

        threading.Thread(target=_thread, daemon=True).start()

    def stop_recording(self):
        self.running = False
        self.record_button.config(text="Начать запись")
        self.recording_indicator.config(text="")
        self.save_recording()

    def save_recording(self):
        if self.recorded_data:
            audio = np.concatenate(self.recorded_data, axis=0)
            filename = datetime.datetime.now().strftime("recording_%Y%m%d_%H%M%S.wav")
            path = os.path.join(self.save_directory, filename)
            audio_int16 = np.int16(audio * 32767)
            write_wav(path, samplerate, audio_int16)
            print(f"Сохранено: {path}")
            self.dir_label.config(text=f"✅ Сохранено: {os.path.basename(path)}")

    def process_audio_from_queue(self):
        while True:
            audio_chunk = self.audio_queue.get()
            print(f"[Обработка] Получено {len(audio_chunk)} отсчётов.")
            # Здесь можно сделать анализ аудио, например, транскрипцию
            # Для примера, добавляем текст
            self.transcription_label.config(text="Здесь появляется транскрибированный текст")

root = tk.Tk()
app = AudioApp(root)
root.mainloop()
