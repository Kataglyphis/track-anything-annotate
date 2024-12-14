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


def mask_map(mask):
    labels = np.unique(mask)
    labels = labels[labels!=0].tolist()
    object_images = []

    for value in labels:
        # Создаем маску для текущего объекта
        object_mask = (mask == value).astype(np.uint8)

        # Создаем черное изображение с теми же размерами, что и маска
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

        # Устанавливаем белый цвет для текущего объекта
        colored_mask[object_mask > 0] = [255, 255, 255]

        # Добавляем изображение объекта в список
        object_images.append(colored_mask)

    return object_images
