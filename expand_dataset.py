import cv2
import numpy as np
import os
from PIL import Image, ImageFilter, ImageEnhance
import random

# 🔹 Пути к папкам
screenshots_path = "letters_screenshots"  # Где лежат скриншоты
dataset_path = "dataset_generated_2"      # Куда добавляем новые изображения

# 🔹 Проверяем существование папок
if not os.path.exists(screenshots_path):
    raise FileNotFoundError(f"❌ Ошибка: Папка {screenshots_path} не найдена!")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Ошибка: Папка {dataset_path} не найдена!")

# 🔹 Параметры генерации
final_size = 32  # Размер изображений для модели
num_new_images = 200  # Сколько новых изображений генерировать для каждой буквы

# 🔹 Функция для уменьшенного шума
def add_salt_pepper_noise(image, amount=0.002):
    """Добавляет минимальный шум Salt & Pepper (чёрные и белые пиксели)"""
    img_np = np.array(image)
    total_pixels = img_np.size
    num_salt = int(amount * total_pixels)  # Количество белых пикселей
    num_pepper = int(amount * total_pixels)  # Количество чёрных пикселей

    # Добавляем белые пиксели (соль)
    for _ in range(num_salt):
        x, y = random.randint(0, img_np.shape[0] - 1), random.randint(0, img_np.shape[1] - 1)
        img_np[x, y] = 255

    # Добавляем чёрные пиксели (перец)
    for _ in range(num_pepper):
        x, y = random.randint(0, img_np.shape[0] - 1), random.randint(0, img_np.shape[1] - 1)
        img_np[x, y] = 0

    return Image.fromarray(img_np)

# 🔹 Обрабатываем каждую букву
for letter in os.listdir(screenshots_path):
    letter_folder = os.path.join(screenshots_path, letter)

    # 🔹 Пропускаем файлы (например, `.DS_Store`)
    if not os.path.isdir(letter_folder):
        continue

    dataset_folder = os.path.join(dataset_path, letter)

    # Если буквы нет в основном датасете – создаём её
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
        print(f"⚠️ Внимание: Создана новая папка {dataset_folder}")

    # Берём все файлы изображений (фильтруем только картинки)
    image_files = [f for f in os.listdir(letter_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"⚠️ Пропускаем букву {letter}, так как нет изображений.")
        continue

    existing_images = len(os.listdir(dataset_folder))  # Количество старых изображений

    # 🔹 Создаём новые изображения на основе скриншотов
    for i in range(num_new_images):
        # Выбираем случайный исходный скриншот
        img_path = os.path.join(letter_folder, random.choice(image_files))
        img = Image.open(img_path).convert("L")  # Чёрно-белое изображение

        # ✅ Приводим к квадратному размеру
        img = img.resize((final_size, final_size))

        # ✅ Слабое размытие (только если случайно выбрано)
        if random.random() > 0.7:
            img = img.filter(ImageFilter.GaussianBlur(0.5))  # Минимальное размытие

        # ✅ Уменьшенные повороты (±15°)
        if random.random() > 0.5:
            img = img.rotate(random.uniform(-15, 15), fillcolor=255)

        # ✅ Добавляем небольшое изменение яркости
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.7, 1.3))  # ±30% яркости

        # ✅ Добавляем минимальный шум (Salt & Pepper)
        if random.random() > 0.5:
            img = add_salt_pepper_noise(img, amount=0.001)  # Меньше шума

        # ✅ Сохраняем изображение в датасет
        file_name = os.path.join(dataset_folder, f"scr_gen_{existing_images + i}.png")
        img.save(file_name)

print("✅ Датасет успешно расширен новыми изображениями!")
