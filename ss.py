import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Пути к модели и меткам
MODEL_PATH = "model.tflite"  # Замените на путь к вашей модели
LABELS_PATH = "labels.txt"   # Замените на путь к файлу с метками

# Загрузка меток
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Инициализация интерпретатора TensorFlow Lite
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Получение входных и выходных тензоров
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Функция для предобработки изображения
def preprocess_image(image):
    input_shape = input_details[0]['shape'][1:3]
    resized_image = cv2.resize(image, input_shape)
    input_data = np.expand_dims(resized_image, axis=0).astype(np.float32)
    input_data = input_data / 255.0  # Нормализация
    return input_data

# Функция для классификации изображения
def classify_image(image):
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    confidence = np.max(output_data)
    return labels[predicted_class], confidence

# Инициализация веб-камеры
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру.")
    exit()

print("Камера успешно открыта. Нажмите 'q' для выхода.")

while True:
    # Захват кадра
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось захватить кадр.")
        break

    # Отображение кадра
    cv2.imshow("Webcam", frame)

    # Распознавание буквы по нажатию клавиши 'r'
    key = cv2.waitKey(1)
    if key == ord('r'):  # Нажмите 'r' для распознавания
        predicted_letter, confidence = classify_image(frame)
        print(f"Распознанная буква: {predicted_letter} | Уверенность: {confidence:.2f}")

    # Выход по нажатию 'q'
    if key == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()