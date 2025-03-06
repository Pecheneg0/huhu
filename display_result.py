import paramiko  # Для SSH-подключения

# 🔹 Подключение к Raspberry Pi
def connect_to_pi(hostname, username, password):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, username=username, password=password)
    return client

# 🔹 Запуск скрипта на Raspberry Pi
def run_script_on_pi(ssh_client):
    stdin, stdout, stderr = ssh_client.exec_command("python3 /home/pi/process_image.py")
    result = stdout.read().decode().strip()
    return result

# 🔹 Основной код
if __name__ == "__main__":
    # Параметры подключения
    hostname = "192.168.1.100"  # Замените на IP-адрес Raspberry Pi
    username = "pi"
    password = "raspberry"

    # Подключаемся к Raspberry Pi
    ssh_client = connect_to_pi(hostname, username, password)

    # Запускаем скрипт на Raspberry Pi и получаем результат
    result = run_script_on_pi(ssh_client)

    # Закрываем соединение
    ssh_client.close()

    # Выводим результат на экран
    print("📌 **Результат распознавания:**")
    print(result)