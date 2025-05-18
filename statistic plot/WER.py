import matplotlib.pyplot as plt
import numpy as np # Для удобства работы с массивами, если понадобится

# --- Данные для каждой конфигурации модели  ---
experiment_data = [
    {
        "label": "Слои LSTM: 3, Скрытые признаки в слое: 1024, Набор данных: GOLOS",
        "epochs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "wer_values": [26.78, 20.81, 17.28, 15.88, 13.81, 12.63, 12.57, 11.24, 10.62, 10.61, 9.72]
    },
    {
        "label": "Слои LSTM: 4, Скрытые признаки в слое: 1024, Набор данных: GOLOS и OPEN STT (phone calls 1,2)",
        "epochs": [1, 2, 3, 4, 5, 6, 7],
        "wer_values": [45.53, 41.92, 38.63, 37.06, 35.48, 34.03, 33.20]
    },
    {
        "label": "Слои LSTM: 7, Скрытые признаки в слое: 1024, Набор данных: GOLOS и OPEN STT (phone calls 1,2)",
        "epochs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        "wer_values": [93.12, 87.56, 78.23, 74.98, 70.50, 68.12, 66.37, 65.84, 64.71, 62.90, 61.32, 60.35, 60.18]
    },
    {
        "label": "Слои LSTM: 4, Скрытые признаки в слое: 1024, Набор данных: GOLOS и OPEN STT (phone calls 1,2)",
        "epochs": [1, 2, 3, 4, 5],
        "wer_values": [75.20, 69.81, 66.33, 63.84, 62.11]
    },
    {
        "label": "Слои LSTM: 4, Скрытые признаки в слое: 768, Набор данных: GOLOS и OPEN STT (phone calls 1,2)",
        "epochs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "wer_values": [87.21, 74.33, 71.58, 65.70, 64.02, 61.76, 61.03, 60.11, 58.76, 56.58, 55.97]
    }
]

# --- Только сверточные слои без LSTM ---
raw_signal_conv = [
        {
        "label": "Модель анализирующая амлитудные отсчеты только через сверточные слои без слоев LSTM",
        "epochs": [1, 2, 3, 4, 5, 6, 7],
        "wer_values": [64, 55, 49, 47, 44, 42, 40]
    }
]

# --- Данные для каждой конфигурации модели с нормализацей до и после ReLU ---
BatchNorm2d_experiment = [
        {
        "label": "Слои LSTM: 3, Скрытые признаки в слое: 1024, Нормализация до ReLU",
        "epochs": [1, 2, 3],
        "wer_values": [26.78, 20.81, 17.28]
    },
        {
        "label": "Слои LSTM: 3, Скрытые признаки в слое: 1024, Нормализация после ReLU",
        "epochs": [1, 2, 3 ],
        "wer_values": [26.41, 20.78, 17.20]
    },
]

# --- Данные для каждой конфигурации модели с разным числом мэл-уровней ---
mel_bins_experiment = [
        {
        "label": "Слои LSTM: 3, Скрытые признаки в слое: 1024, Мэл-уровней: 80",
        "epochs": [1, 2, 3, 4, 5, 6],
        "wer_values": [26.78, 20.81, 17.28, 15.88, 13.81, 12.63]
    },
        {
        "label": "Слои LSTM: 3, Скрытые признаки в слое: 1024, Мэл-уровней: 128",
        "epochs": [1, 2, 3, 4, 5, 6],
        "wer_values": [28.23, 22.35, 17.64, 16.76, 14.14, 13.72]
    },
]

# --- Построение графика ---
plt.figure(figsize=(16, 9)) # Увеличим для читаемости подписей

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
linestyles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '^', 'v', '<', '>']

#for i, data in enumerate(experiment_data):
for i, data in enumerate(raw_signal_conv):
    plt.plot(data["epochs"], data["wer_values"],
             marker=markers[i % len(markers)],
             linestyle=linestyles[i % len(linestyles)],
             color=colors[i % len(colors)],
             linewidth=2,
             markersize=7,
             label=data["label"])

    # ---> ДОБАВЛЯЕМ ПОДПИСИ ЗНАЧЕНИЙ К ТОЧКАМ <---
    for epoch_val, wer_val in zip(data["epochs"], data["wer_values"]):
        # Добавляем небольшое смещение по Y, чтобы текст был чуть выше точки
        # Выравнивание по горизонтали 'center'
        # Размер шрифта можно уменьшить, если надписи накладываются
        plt.annotate(f"{wer_val:.2f}", # Текст подписи (WER с 2 знаками после запятой)
                     (epoch_val, wer_val),          # Координаты точки (x,y)
                     textcoords="offset points",    # Система координат для смещения текста
                     xytext=(0, 8),                 # Смещение текста (0 по X, 8 пикселей вверх по Y)
                     ha='center',                   # Горизонтальное выравнивание текста
                     fontsize=8,                    # Размер шрифта подписи
                     color=colors[i % len(colors)]) # Можно использовать цвет линии для подписи

# Настройка графика
plt.title('Сравнение динамики Validation WER', fontsize=16)
plt.xlabel('Эпоха', fontsize=14)
plt.ylabel('Word Error Rate (WER) %', fontsize=14)

max_epoch = 0
for data in raw_signal_conv:
    if data["epochs"]:
        current_max = max(data["epochs"])
        if current_max > max_epoch:
            max_epoch = current_max

if max_epoch > 0:
    # Показываем каждую эпоху, если их не слишком много, иначе можно увеличить шаг
    step = 1 if max_epoch <= 20 else int(np.ceil(max_epoch / 20)) # Не более ~20 меток
    plt.xticks(np.arange(1, max_epoch + 1, step=step), fontsize=12)
else:
    plt.xticks(fontsize=12)

plt.yticks(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11, title='Конфигурация модели', title_fontsize=12, loc='upper right')
plt.tight_layout(pad=1.5) # Добавим немного отступа для подписей

# Сохранение графика в файл
plt.savefig('comparison_mel_bin_plot.png', dpi=150) # Увеличим dpi для лучшего качества
print("График сохранен в comparison_wer_plot_with_values.png")

# Отображение графика
plt.show()