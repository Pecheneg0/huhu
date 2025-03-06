import cv2
import numpy as np

# Параметры
FPS = 15  # Частота кадров
ZOOM_FACTOR = 2.0  # Коэффициент цифрового зума

# Инициализация камеры
cap = cv2.VideoCapture(0)

# Проверка, открыта ли камера
if not cap.isOpened():
    print("Ошибка: Камера не подключена или не найдена.")
    exit()

# Установка частоты кадров
cap.set(cv2.CAP_PROP_FPS, FPS)

# Получение размеров кадра
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Разрешение камеры: {width}x{height}")

# Функция для применения цифрового зума
def apply_zoom(frame, zoom_factor):
    h, w = frame.shape[:2]
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    start_x, start_y = (w - new_w) // 2, (h - new_h) // 2
    zoomed_frame = frame[start_y:start_y + new_h, start_x:start_x + new_w]
    return cv2.resize(zoomed_frame, (w, h), zoom_factor)

# Основной цикл
while True:
    # Захват кадра
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось захватить кадр.")
        break

    # Применение цифрового зума
    zoomed_frame, zoom_factor = apply_zoom(frame, ZOOM_FACTOR)

    # Вывод информации о зуме
    cv2.putText(zoomed_frame, f"Zoom: x{zoom_factor}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Отображение кадра
    cv2.imshow("Camera Stream with Zoom", zoomed_frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()