import cv2
import numpy as np
import os
from PIL import Image, ImageFilter
import random

# 🔹 Папки с изображениями
screenshots_path = "letters_screenshots"  # Где лежат скриншоты
dataset_path = "dataset_generated_2"      # Куда их добавлять

# 🔹 Проверяем существование папок
if not os.path.exists(screenshots_path):
    raise FileNotFoundError(f"❌ Ошибка: Папка {screenshots_path} не найдена!")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Ошибка: Папка {dataset_path} не найдена!")

# 🔹 Размеры изображений
final_size = 32  # Размер для нейросети

# 🔹 Обрабатываем каждую букву
for letter in os.listdir(screenshots_path):
    letter_folder = os.path.join(screenshots_path, letter)
    dataset_folder = os.path.join(dataset_path, letter)

    # Если буквы нет в основном датасете – создаём её
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
        print(f"⚠️ Внимание: Создана новая папка {dataset_folder}")

    # Берём все файлы изображений
    image_files = [f for f in os.listdir(letter_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    existing_images = len(os.listdir(dataset_folder))  # Количество старых изображений

    # 🔹 Обрабатываем каждое изображение
    for i, file in enumerate(image_files):
        img_path = os.path.join(letter_folder, file)
        img = Image.open(img_path).convert("L")  # Чёрно-белое изображение

        # ✅ Приводим к квадратному размеру (обрезаем или дополняем)
        img = img.resize((final_size, final_size))

        # ✅ Добавляем случайные аугментации
        if random.random() > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0, 1)))  # Размытие
        if random.random() > 0.5:
            img = img.rotate(random.uniform(-15, 15), fillcolor=255)  # Поворот

        # ✅ Сохраняем изображение в датасет
        file_name = os.path.join(dataset_folder, f"scr_{existing_images + i}.png")
        img.save(file_name)

print("✅ Скриншоты успешно добавлены в датасет!")
