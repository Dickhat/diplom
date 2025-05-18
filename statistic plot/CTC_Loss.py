import matplotlib.pyplot as plt

# Ваши данные Avg Train Loss по эпохам
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Номера эпох
avg_valid_loss_values = [0.4848, 0.2449, 0.1921, 0.1596, 0.1443, 0.1304, 0.1226, 0.1183, 0.1063, 0.1046, 0.0998]  # Значения Avg Train Loss

# --- Построение графика ---
plt.figure(figsize=(10, 6))  # Задаем размер фигуры

# Строим линию графика
plt.plot(epochs, avg_valid_loss_values, marker='s', linestyle='-', color='g', label='Avg Valid Loss') # 's' - квадратный маркер, 'g' - зеленый цвет

# Добавляем маркеры с точными значениями Loss на графике (опционально)
for i, txt in enumerate(avg_valid_loss_values):
    plt.annotate(f"{txt:.4f}", (epochs[i], avg_valid_loss_values[i]), textcoords="offset points", xytext=(0,5), ha='center')

# Настройка графика
plt.title('Динамика Avg CTC Loss по эпохам')  # Заголовок графика
plt.xlabel('Эпоха')                              # Подпись оси X
plt.ylabel('Средний CTC Loss валидации')         # Подпись оси Y
plt.xticks(epochs)                               # Метки для каждой эпохи на оси X
plt.grid(True, linestyle='--', alpha=0.7)        # Добавляем сетку
plt.legend()                                     # Отображаем легенду
plt.tight_layout()                               # Автоматически подгоняет элементы

# Сохранение графика в файл (опционально)
plt.savefig('avg_valid_loss_over_epochs_plot.png')
print("График сохранен в avg_valid_loss_over_epochs_plot.png")

# Отображение графика
plt.show()