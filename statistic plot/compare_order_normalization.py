import matplotlib.pyplot as plt
import numpy as np

# Данные для экспериментов
BatchNorm2d_experiment = [
    {
        "label": "Слои LSTM: 3, Нейроны: 1024, BN до ReLU", # Сократил для легенды
        "epochs": [1, 2, 3],
        "wer_values": [26.78, 20.81, 17.28]
    },
    {
        "label": "Слои LSTM: 3, Нейроны: 1024, BN после ReLU", # Сократил для легенды
        "epochs": [1, 2, 3],
        "wer_values": [26.41, 20.78, 17.20]
    },
]

# --- Построение двух отдельных графиков в одном окне ---
# Создаем фигуру и два набора осей (subplot'а)
# fig - это вся область окна
# axs - это массив объектов осей, в данном случае 2 строки, 1 столбец
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

# Цвета и стили
colors = ['dodgerblue', 'orangered']
linestyles = ['-', '--']
markers = ['o', 's']

# --- График 1: BN до ReLU ---
data1 = BatchNorm2d_experiment[0]
axs[0].plot(data1["epochs"], data1["wer_values"],
            marker=markers[0],
            linestyle=linestyles[0],
            color=colors[0],
            linewidth=2,
            markersize=7,
            label=data1["label"])

# Добавляем подписи значений к точкам для первого графика
for epoch_val, wer_val in zip(data1["epochs"], data1["wer_values"]):
    axs[0].annotate(f"{wer_val:.2f}%",
                    (epoch_val, wer_val),
                    textcoords="offset points",
                    xytext=(0, 8), # Смещение вверх
                    ha='center',
                    fontsize=9,
                    color=colors[0])

axs[0].set_title('Нормализация до ReLU', fontsize=14)
axs[0].set_ylabel('Word Error Rate (WER) %', fontsize=12)
axs[0].set_xlabel('Эпоха', fontsize=12) # Подпись оси X будет общей
axs[0].grid(True, linestyle=':', alpha=0.6)
axs[0].legend(fontsize=10, loc='upper right')
axs[0].set_xticks(data1["epochs"]) # Метки для каждой эпохи на оси X
axs[0].tick_params(axis='y', labelsize=10)
axs[0].tick_params(axis='x', labelsize=10)


# --- График 2: BN после ReLU ---
data2 = BatchNorm2d_experiment[1]
axs[1].plot(data2["epochs"], data2["wer_values"],
            marker=markers[1],
            linestyle=linestyles[1],
            color=colors[1],
            linewidth=2,
            markersize=7,
            label=data2["label"])

# Добавляем подписи значений к точкам для второго графика
for epoch_val, wer_val in zip(data2["epochs"], data2["wer_values"]):
    axs[1].annotate(f"{wer_val:.2f}%",
                    (epoch_val, wer_val),
                    textcoords="offset points",
                    xytext=(0, 8), # Смещение вверх
                    ha='center',
                    fontsize=9,
                    color=colors[1])

axs[1].set_title('Нормализация после ReLU', fontsize=14)
axs[1].set_ylabel('Word Error Rate (WER) %', fontsize=12)
axs[1].set_xlabel('Эпоха', fontsize=12) # Общая подпись оси X для нижнего графика
axs[1].grid(True, linestyle=':', alpha=0.6)
axs[1].legend(fontsize=10, loc='upper right')
axs[1].set_xticks(data2["epochs"]) # Метки для каждой эпохи на оси X (хотя ось общая)
axs[1].tick_params(axis='y', labelsize=10)
axs[1].tick_params(axis='x', labelsize=10)


# Общий заголовок для всего окна
fig.suptitle('Сравнение влияния расположения BatchNorm2d на Validation WER', fontsize=16, y=0.99) # y для небольшого смещения

# Автоматически подгоняет элементы графика, чтобы они не перекрывались
# и добавляет немного пространства между подграфиками
fig.tight_layout(rect=[0, 0, 1, 0.97]) # rect для учета suptitle

# Сохранение графика в файл
plt.savefig('bn_relu_order_comparison_plot.png', dpi=150)
print("График сохранен в bn_relu_order_comparison_plot.png")

# Отображение графика
plt.show()