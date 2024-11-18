import requests

# URL файла для скачивания
url = 'https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth'
# Имя файла, под которым он будет сохранен
file_name = 'models/XMem.pth'

# Выполняем запрос на скачивание
response = requests.get(url)

# Проверяем, успешен ли запрос
if response.status_code == 200:
    # Открываем файл в бинарном режиме и записываем содержимое
    with open(file_name, 'wb') as file:
        file.write(response.content)
    print(f"Файл '{file_name}' успешно скачан.")
else:
    print(f"Ошибка при скачивании файла: {response.status_code}")
