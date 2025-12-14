### shark-species-classifier

Этот репозиторий содержит код для проекта **классификации видов акул по фотографии**
(13 классов).

- **Вход**: RGB‑изображение (JPG/PNG)
- **Выход**: вероятности по 13 классам (softmax)
- **Метрики**: Accuracy, Macro F1
- **Данные**: Kaggle dataset “Shark Species” (`https://www.kaggle.com/datasets/larusso94/shark-species/data`)

---

### Setup

1. Установить зависимости (uv):

```bash
uv sync --extra dev
```

2. Установить хуки:

```bash
uv run pre-commit install
```

3. Проверить качество кода:

```bash
uv run pre-commit run -a
```

---

### Train

Опционально: поднять MLflow server (по умолчанию проект ждёт `http://127.0.0.1:8080`):

```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

Запуск обучения:

```bash
uv run python -m shark_species_classifier.commands command=train
```

Пример override параметров:

```bash
uv run python -m shark_species_classifier.commands command=train trainer.max_epochs=10
```

---

### Data (DVC / download)

Данные **не хранятся в git**.

При запуске `train`/`infer` код:

- пробует `dvc pull`
- если DVC не настроен/не сработал — скачивает архив по публичной ссылке из `configs/data/default.yaml`
  и распаковывает в `data.raw_dir`

---

### Infer

Инференс по одному изображению (чекпойнт можно не указывать — возьмётся самый новый из `checkpoints/`):

```bash
uv run python -m shark_species_classifier.commands command=infer infer.image_path=path/to/image.jpg
```
