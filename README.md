Для запуска скрипта "for_rp4"
Установить библиотеки

sudo apt update
sudo apt upgrade -y
sudo apt install python3-opencv python3-numpy
pip3 install tflite-runtime




Для проверки работы камеры 

-включить поддержку камеры
sudo raspi-config

-перезапустить
sudo reboot

-Установка библиотек
sudo apt update
sudo apt install python3-opencv
-запустить скрипт test_camera

Если возникла ошибка ilbcamera:
1) sudo apt install libcamera-dev
2) libcamera_hello


проверка зума (test_zoom)
Если герцовка проседает, стоит уменьшить разрешение. 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)



для основного  алгоритма :
Объяснение работы скрипта

Захват изображения:
Используется камера Raspberry Pi для захвата изображения.
Поиск квадратного контура:
Изображение преобразуется в черно-белое, и применяется пороговая обработка для выделения белых областей.
Находится контур с 4 углами, который близок к квадрату.
Выравнивание контура:
Если контур повернут, применяется перспективное преобразование для выравнивания.
Классификация изображения:
Выровненное изображение передается в модель TensorFlow Lite для распознавания буквы.
Вывод результата:
Результат распознавания выводится в терминал, а выровненное изображение отображается в окне.


