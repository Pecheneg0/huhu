import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Пути к модели и меткам
MODEL_PATH = "model.tflite"
LABELS_PATH = "labels.txt"

# Загрузка меток
with open(LABELS_PATH, "r") as f:
    labels = [line.strip().split(" ")[1] for line in f.readlines()]

# Инициализация интерпретатора TensorFlow Lite
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Получение входных и выходных тензоров
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Функция для захвата изображения с камеры
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: Камера не подключена. Код ошибки: 500")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Ошибка: Не удалось захватить изображение. Код ошибки: 501")
        return None
    return frame

# Функция для обработки изображения
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.9 < aspect_ratio < 1.1:  # Проверка на квадратность
                roi = image[y:y+h, x:x+w]
                return roi, (x, y, w, h)
    return None, None

# Функция для классификации изображения
def classify_image(image):
    input_shape = input_details[0]['shape'][1:3]
    resized_image = cv2.resize(image, input_shape)
    input_data = np.expand_dims(resized_image, axis=0).astype(np.float32)
    input_data = input_data / 255.0  # Нормализация

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    confidence = np.max(output_data)
    return labels[predicted_class], confidence

# Основной цикл
def main():
    while True:
        command = input("Нажмите Enter для захвата изображения или 'q' для выхода: ")
        if command.lower() == 'q':
            break

        image = capture_image()
        if image is None:
            continue

        roi, bbox = preprocess_image(image)
        if roi is None:
            print("Ошибка: Квадратный объект не найден. Код ошибки: 404")
            continue

        predicted_letter, confidence = classify_image(roi)
        if confidence < 0.5:  # Порог уверенности
            print ("Предупреждение: Буква не распознана с достаточной уверенностью ")