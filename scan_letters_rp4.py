import cv2
import torch
import numpy as np
import time
from picamera2 import Picamera2
import math
import os

# 🔹 Загрузка модели
from model import ArmenianLetterNet

model = ArmenianLetterNet()
model.load_state_dict(torch.load("armenian_letters_finetuned.pth", map_location="cpu"))
model.eval()

# 🔹 Параметры системы
HEIGHT = 35  # Высота полёта (30-40 м)
CAMERA_ANGLE = 45  # Угол наклона камеры (градусы)
FOV_H = 62.2  # Горизонтальный угол обзора камеры
FOV_V = 48.8  # Вертикальный угол обзора камеры
RESOLUTION = (3280, 2464)  # Разрешение камеры

# 🔹 Фокусное расстояние камеры (в метрах)
FOCAL_LENGTH = HEIGHT * math.tan(math.radians(FOV_V / 2))

# 🔹 Настройка камеры
camera = Picamera2()
camera.preview_configuration.main.size = RESOLUTION
camera.preview_configuration.main.format = "RGB888"
camera.configure("preview")
camera.start()

# 🔹 Фильтр предсказаний
confidence_threshold_low = 0.75   # 📌 Минимальная уверенность для учёта предсказания
confidence_threshold_high = 0.85  # 📌 Уверенность, при которой сразу фиксируем результат
frame_buffer = []  # Буфер предсказаний для усреднения

# 🔹 Функция обработки изображения
def process_image(image):
    """ Обрабатывает изображение для распознавания букв """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Чёрно-белое
    resized = cv2.resize(gray, (32, 32))  # Приведение к размеру модели
    tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    return tensor

# 🔹 Функция перевода пикселей в координаты
def pixel_to_coordinates(px, py, drone_lat, drone_lon):
    """ Переводит пиксели буквы в реальные координаты """
    cx, cy = RESOLUTION[0] // 2, RESOLUTION[1] // 2  # Центр изображения

    # 🔹 Метры на пиксель
    meters_per_pixel = (2 * HEIGHT * math.tan(math.radians(FOV_H / 2))) / RESOLUTION[0]

    # 🔹 Смещение буквы относительно центра камеры
    dx = (px - cx) * meters_per_pixel
    dy = (py - cy) * meters_per_pixel * math.cos(math.radians(CAMERA_ANGLE))

    # 🔹 Вычисление реальных координат
    lat_offset = dy / 111320  # 1 градус = ~111.32 км
    lon_offset = dx / (111320 * math.cos(math.radians(drone_lat)))

    return drone_lat + lat_offset, drone_lon + lon_offset

# 🔹 Функция сохранения координат
def save_coordinates(letter, lat, lon, confidence):
    """ Записывает найденные буквы и координаты на SD-карту """
    with open("/home/pi/letters_coordinates.txt", "a") as file:
        file.write(f"{letter},{lat},{lon},{confidence:.2f}\n")

# 🔹 Основной цикл
def scan_letters():
    """ Сканирование букв и запись координат """
    global frame_buffer

    while True:
        frame = camera.capture_array()  # 📷 Фотографируем
        tensor_image = process_image(frame)  # 🖼 Обрабатываем изображение
        
        with torch.no_grad():
            output = model(tensor_image)  # 🔍 Распознаём букву
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        confidence_value = confidence.item()
        letter = chr(1329 + predicted_class.item())

        # 🔹 Фильтрация по уверенности
        if confidence_value < confidence_threshold_low:
            print(f"⚠️ Слабое предсказание ({confidence_value:.2f}) - игнорируем")
            continue  # Пропускаем слабые предсказания

        # 🔹 Если уверенность 75-85%, проверяем несколько кадров
        if confidence_threshold_low <= confidence_value < confidence_threshold_high:
            frame_buffer.append((letter, confidence_value))

            if len(frame_buffer) >= 5:  # Анализируем 5 кадров подряд
                avg_confidence = sum([c[1] for c in frame_buffer]) / len(frame_buffer)
                most_common_letter = max(set([c[0] for c in frame_buffer]), key=[c[0] for c in frame_buffer].count)
                
                if avg_confidence >= confidence_threshold_high:
                    print(f"✅ Надёжное предсказание ({most_common_letter}, {avg_confidence:.2f})")
                    frame_buffer.clear()  # Очищаем буфер
                else:
                    print(f"⚠️ Неопределённый результат ({avg_confidence:.2f}) - пропускаем")
                    frame_buffer.clear()
                    continue  # Пропускаем ненадёжные предсказания

        # 🔹 Если уверенность выше 85%, записываем немедленно
        if confidence_value >= confidence_threshold_high:
            drone_lat, drone_lon = 40.1792, 44.4991  # 🔹 Заглушка, заменить на GPS!
            letter_lat, letter_lon = pixel_to_coordinates(RESOLUTION[0]//2, RESOLUTION[1]//2, drone_lat, drone_lon)

            save_coordinates(letter, letter_lat, letter_lon, confidence_value)
            print(f"✅ Найдена буква {letter} ({confidence_value:.2f}) на координатах {letter_lat}, {letter_lon}")

        # 🔹 Энергосбережение – пауза между сканированием
        time.sleep(2)  

# 🚀 Запускаем сканирование
scan_letters()
