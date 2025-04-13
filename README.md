- Установка через uv
```bash
# для CUDA
uv sync --extra cu124
# для CPU
uv sync --extra cpu
```

- Скачивание моделей
```bash
uv run checkpoints\download_models.py
```
- Запуск демо http://127.0.0.1:8080
```bash
gradio demo.py
```
![alt text](video-test\cache\image.png)

- Для создания датасетов 
```bash
uv run annotation.py
```
---
- Установка pytorch для CUDA
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

- Установка остальных зависемостей
```bash
pip install -r requirements.txt
```