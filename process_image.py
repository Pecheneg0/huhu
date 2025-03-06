import torch
import cv2
import numpy as np
from model import ArmenianLetterNet  # Импортируем модель

# 🔹 Порог уверенности
confidence_threshold = 0.55

# 🔹 Армянский алфавит (от Юникода 1329 до 1366)
armenian_alphabet = [chr(code) for code in range(0x0531, 0x0557)]

# 🔹 Загружаем модель
model = ArmenianLetterNet()
model.load_state_dict(torch.load("armenian_letters_model_2.pth", map_location="cpu"))
model.eval()

# 🔹 Функция для захвата и анализа изображения
def capture_and_analyze():
    # Захват изображения с камеры
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Ошибка: Камера не подключена."

    ret, frame = cap.read()
    if not ret:
        return "Ошибка: Не удалось захватить изображение."

    # Подготовка изображения
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (32, 32))
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0  # Нормализация

    # Предсказание модели
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    confidence_value = confidence.item()
    predicted_index = predicted_class.item()

    # Проверяем, предсказан ли корректный индекс
    if 0 <= predicted_index < len(armenian_alphabet):
        predicted_letter = armenian_alphabet[predicted_index]
    else:
        predicted_letter = "❌ Ошибка индекса"

    # Формируем результат
    if confidence_value >= confidence_threshold:
        result = f"✅ Распознанная буква: {predicted_letter} | Уверенность: {confidence_value:.2f}"
    else:
        result = f"⚠️ Буква не распознана с достаточной уверенностью! ({confidence_value:.2f})"

    cap.release()
    return result

# 🔹 Основной код
if __name__ == "__main__":
    result = capture_and_analyze()
    print(result)  # Вывод результата в консоль