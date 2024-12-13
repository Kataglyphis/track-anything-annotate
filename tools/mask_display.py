import numpy as np


def visualize_unique_mask(unique_mask):
    # Создаем цветовую карту для визуализации уникальной маски
    unique_values = np.unique(unique_mask)
    color_map = np.zeros((len(unique_values), 3), dtype=np.uint8)

    # Генерация уникальных цветов для каждого уникального значения
    for i, value in enumerate(unique_values):
        if value == 0:  # Фон
            color_map[i] = [0, 0, 0]  # Черный для фона
        else:
            color_map[i] = tuple(
                np.random.randint(0, 255, size=3)
            )  # Случайный цвет для объектов

    # Создаем цветную визуализацию уникальной маски
    colored_unique_mask = np.zeros(
        (unique_mask.shape[0], unique_mask.shape[1], 3), dtype=np.uint8
    )
    for i in range(unique_mask.shape[0]):
        for j in range(unique_mask.shape[1]):
            colored_unique_mask[i, j] = color_map[unique_mask[i, j]]

    return colored_unique_mask


def visualize_wb_mask(mask):
    colored_mask = np.zeros(
        (mask.shape[0], mask.shape[1], 3), dtype=np.uint8
    )  # Фон черный

    # Устанавливаем белый цвет для всех объектов (не нулевых значений)
    colored_mask[mask > 0] = [255, 255, 255]  # Белый цвет для объектов

    return colored_mask
