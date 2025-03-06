import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

# 🔹 Путь к существующему датасету
dataset_path = "dataset_generated_2"

# Проверяем, существует ли папка
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Ошибка: Папка {dataset_path} не найдена!")

# 🔹 Армянский алфавит (буквы от U+0531 до U+0556)
armenian_letters = [chr(code) for code in range(0x0531, 0x0557)]

# 🔹 Размеры изображений
img_size = 128   # Оригинальный размер
final_size = 32  # Размер для нейросети

# 🔹 Количество новых изображений для каждой буквы
num_new_images = 200  # Можно изменить

# 🔹 Проверяем существующие папки
for letter in armenian_letters:
    letter_folder = os.path.join(dataset_path, letter)
    
    if not os.path.exists(letter_folder):
        print(f"⚠️ Внимание: Папка {letter} отсутствует. Создаю...")
        os.makedirs(letter_folder)  # Создаём папку, если её нет

    # Добавляем новые изображения
    existing_images = len(os.listdir(letter_folder))  # Сколько уже есть изображений

    for i in range(num_new_images):
        img = Image.new('L', (img_size, img_size), 255)  # Белый фон
        draw = ImageDraw.Draw(img)

        # 🔹 Выбираем случайный шрифт (добавьте свои армянские шрифты)
        font_path = random.choice([
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
        ])  
        font = ImageFont.truetype(font_path, random.randint(80, 120))

        # 📌 Определяем размеры текста
        bbox = draw.textbbox((0, 0), letter, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((img_size - w) // 2, (img_size - h) // 2), letter, font=font, fill=0)

        # ✅ Добавляем случайный поворот
        img = img.rotate(random.uniform(-20, 20), fillcolor=255)

        # ✅ Размытие (имитация вибрации дрона)
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0, 1)))

        # ✅ Добавляем шум
        noise = np.random.normal(0, 15, (img_size, img_size))
        img_np = np.array(img) + noise
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)

        # ✅ Уменьшаем до 32x32 (размер модели)
        img = img.resize((final_size, final_size))

        # ✅ Сохраняем файл
        file_name = os.path.join(letter_folder, f"new_{existing_images + i}.png")
        img.save(file_name)

print(f"✅ Датасет {dataset_path} успешно расширен!")
