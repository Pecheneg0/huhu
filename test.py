import torch
import cv2
import numpy as np
from model import ArmenianLetterNet

# Загружаем модель
model = ArmenianLetterNet()
model.load_state_dict(torch.load("armenian_letters_model.pth", map_location="cpu"))
model.eval()

# Тестируем на одном изображении
image = cv2.imread("dataset_generated_1/Ւ/9.png", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (32, 32))
image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

with torch.no_grad():
    output = model(image)
    _, predicted_class = torch.max(output, 1)

print(f"🔤 Распознанная буква: {chr(1329 + predicted_class.item())}")
