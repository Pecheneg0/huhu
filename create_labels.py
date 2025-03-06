# Армянский алфавит (Unicode коды от 0x0531 до 0x0556)
armenian_alphabet = [chr(code) for code in range(0x0531, 0x0557)]

# Создание файла labels.txt
with open("labels.txt", "w", encoding="utf-8") as f:
    for index, letter in enumerate(armenian_alphabet):
        f.write(f"{index} {letter}\n")

print("Файл labels.txt успешно создан.")