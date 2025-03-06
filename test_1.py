import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import ArmenianLetterNet  # Импортируем модель

# 🔹 Порог уверенности
confidence_threshold = 0.75

# 🔹 Загружаем модель
model = ArmenianLetterNet()
model.load_state_dict(torch.load("armenian_letters_model.pth", map_location="cpu"))
model.eval()

# 🔹 Загружаем тестовое изображение
image_path = "dataset_generated_1/Ա/0.png"  # Замените на путь к тестовому изображению
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (32, 32))
image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

# 🔹 Предсказание модели
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probabilities, 1)

confidence_value = confidence.item()

# 🔹 Определяем результат
if confidence_value >= confidence_threshold:
    recognized_letter = chr(1329 + predicted_class.item())  # Армянские буквы начинаются с Юникода 1329
    text_color = "green"
    result_text = f"✅ Распознанная буква: {recognized_letter} (Уверенность: {confidence_value:.2f})"
else:
    recognized_letter = "❓"  # Символ неизвестной буквы
    text_color = "red"
    result_text = "⚠️ Буква не распознана с достаточной уверенностью!"

# 🔹 Визуализация результата
image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Преобразуем в цветное изображение
plt.figure(figsize=(5, 5))
plt.imshow(image_rgb, cmap="gray")
plt.axis("off")
plt.title(result_text, color=text_color, fontsize=12, weight="bold")
plt.show()

# 🔹 Вывод в консоль
print(result_text)
print(f"🔢 Индекс буквы: {predicted_class.item()} | 📊 Уверенность: {confidence_value:.2f}")
