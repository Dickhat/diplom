import torch 
import torch.nn as nn 
import torchaudio
import torchaudio.transforms as transforms
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration

def transcribe(file_path):
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")


    waveform, orig_sr = librosa.load(file_path, sr=16000, mono=True)

    # Use the model and processor to transcribe the audio:
    input_features = processor(waveform, sampling_rate=orig_sr, return_tensors="pt").input_features

    # Generate token ids
    predicted_ids = model.generate(input_features, language="russian")

    # Decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0]

class Encoder(nn.Module):
    ''' Класс Encoder нейронной модели '''
    def __init__(self, 
                 num_mels=80, # Число мэл уровней
                 d_model=512,  # Число скрытых признаков в слое
                 num_layers=6, # Число слоев (повторяющихся блоков Encoder)
                 n_heads=8,    # Число голов-внимания
                 ff_dim=2048,  # feed-forward network - полносвязный слой после голов-внимания
                 dropout=0.1   # Вероятность зануления нейронов
                 ): 
        super().__init__()
        
        # Линейное преобразование мел-спектрограммы в скрытое пространство
        self.input_proj = nn.Linear(num_mels, d_model)
        
        # Представление слоя Encoder в Трансформере  
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) # Создание Encoder с num_layers слоями

        # Позиционные эмбеддинги содержат информацию о позиции каждого токена в последовательности
        self.pos_embedding = nn.Parameter(torch.randn(1,        # Количество примеров в батче
                                                      1000,     # max_len = 1000 (максимальная длина последовательности)
                                                      d_model)) # размерность эмбеддингов 

    def forward(self, mel_spec):
        """
        Вход: mel_spec (batch, time_steps, n_mels)
        Выход: скрытые признаки (batch, time_steps, d_model)
        """
        
        # Возможно надо добавить проход по циклу time_steps, чтобы преобразовать каждый шаг мэл-спектраграммы в 512-численные скрытые признаки

        batch_size, num_mels, time_steps = mel_spec.shape
        mel_spec = mel_spec.permute(0, 2, 1).reshape(-1, num_mels)
        #mel_spec = mel_spec.permute(1, 0).reshape(-1, num_mels)

        x = self.input_proj(mel_spec)                   # (batch, time_steps, d_model) преобразуем к num_mels*d_model
        
        x = x.reshape(batch_size, time_steps, -1)

        x = x + self.pos_embedding[:, :x.shape[1], :]   # Добавляем позиционные эмбеддинги (:x.shape[1] - номер временного шага)
        x = self.encoder(x)                             # Проход Encoder-а
        return x

class Decoder(nn.Module):
    def __init__(self, 
                 vocab_size,    # Размер словаря
                 d_model=512,   # Размер скрытого пространства
                 num_layers=6,  # Количество слоев Decoder
                 n_heads=8,     # Количество голов в механизме внимания
                 ff_dim=2048,   # Размер Feed-forward слоя
                 dropout=0.1    # Дропаут
                 ):
        super().__init__()

        # Линейный слой для преобразования признаков Decoder в слова
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Слой для генерации маски, предотвращающей информацию о будущих токенах (для autoregressive).
        self.mask = None

        # Представление слоя Decoder в Трансформере  
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout
        )
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)  # Создание Decoder с num_layers слоями

    def forward(self, memory, tgt, tgt_mask=None):
        """
        Вход:
        - extracted_features: скрытые признаки, полученные от Encoder (batch_size, time_steps, d_model)
        - tgt: целевая последовательность (batch_size, target_length), токены текста на вход (например, пустой начальный токен)

        Выход:
        - выходные токены (batch_size, target_length, vocab_size)
        """
        # Переводим целевые токены в эмбеддинги
        # Здесь предполагается, что tgt - это индексы токенов, и вам нужно их преобразовать в эмбеддинги
        tgt_emb = self.embedding(tgt)  # Если у вас есть слой embedding
        
        # Применяем Transformer Decoder
        # Нужно учитывать, что память (memory) передается из Encoder
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)

        # Получаем финальные логиты через линейный слой
        logits = self.output_proj(output)  # (batch_size, target_length, vocab_size)
        
        return logits

def extract_features(audio_path, sample_rate=16000, print_fig = False):
    ''' Извлечение Мэл-спектрограммы из аудио-файла '''

    # Добавить выравнивание частоты дискретизации до 16 КГц, если исходная частота не равна ей

    # Загружаем аудио
    waveform, orig_sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    # Создаем преобразование в мел-спектрограмму
    mel_spectogram = librosa.feature.melspectrogram(y=waveform, sr=orig_sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(mel_spectogram, ref=np.max) # 10 * log10(S / ref)

    # Отображать лог-мэл-спектрограмму?
    if print_fig:
        plt.figure().set_figwidth(12)
        librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=orig_sr, fmax=8000)
        plt.colorbar()
        plt.show()

    min_value = np.min(S_dB)

    # Добавляем нулевые фреймы (до 3000)
    if S_dB.shape[1] < 3000:
        pad_width = 3000 - S_dB.shape[1]
        S_dB = np.pad(S_dB, ((0, 0), (0, pad_width)), mode='constant', constant_values=min_value)

    mel_spec = np.expand_dims(S_dB, axis=0)  # добавление размерности для batch_size = 1

    return torch.tensor(mel_spec, dtype=torch.float32) # (1, n_mels, time_steps)

# Пример использования
#audio_path = "F:/asr_public_phone_calls_1/0/5c646233c806.opus"  # Замените на путь к вашему файлу
#mel_spec = extract_features(audio_path)

#encoder = Encoder()

#features = encoder(mel_spec)  # Выходная размерность: (batch, time_steps, d_model)
#print("Размерность скрытых признаков:", features.shape)  # Должно быть (1, 300, 512)

# processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")

# predicted_ids = model.generate(mel_spec, language="russian")

# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
# transcription[0]

#text = transcribe("F:/asr_public_phone_calls_1/0/4bc84bbfa234.opus")

feauture_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language = "russian")

waveform, orig_sr = librosa.load("F:/asr_public_phone_calls_1/0/4bc84bbfa234.opus", sr=16000, mono=True)

feature = feauture_extractor(waveform, return_tensors="pt").input_features

generated_tokens = model.generate(input_features=feature)

transcription = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

print(transcription)