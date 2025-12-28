### shark-species-classifier

Этот репозиторий содержит код для проекта **классификации видов акул по фотографии**.

- **Задача**: многоклассовая классификация изображений
- **Количество классов**: 13
- **Вход**: RGB‑изображение (JPG/PNG)
- **Выход**: вероятности по классам (softmax)
- **Метрики**: Accuracy, Macro F1
- **Датасет**: Kaggle “Shark Species” (`https://www.kaggle.com/datasets/larusso94/shark-species/data`)

В проекте используются:

- **PyTorch Lightning**: обучение
- **Hydra**: конфиги (гиперпараметры без “магических констант”)
- **MLflow**: логирование метрик/лоссов/гиперпараметров + тег `git_commit`
- **DVC**: хранение данных (данные не коммитятся в git)

---

### Setup

Требования: установленный [`uv`](https://github.com/astral-sh/uv) и Git.

1. Установить Python подходящей версии (проект рассчитан на **Python 3.10–3.12**):

```bash
uv python install 3.12
uv venv --python 3.12
```

2. Установить зависимости:

```bash
uv sync --extra dev
```

3. Установить и прогнать хуки качества кода:

```bash
uv run pre-commit install
uv run pre-commit run -a
```

---

### Data (DVC / download)

Данные **не хранятся в git**.


При запуске `train`/`infer` код пытается обеспечить наличие данных:

- сначала делает **`dvc pull`** через Python API (если настроен DVC remote)
- если это не сработало и в конфиге задан `data.yandex_public_url`, скачивает zip‑архив и распаковывает в `raw_dir`

Если у вас **не настроен DVC remote**, укажите ссылку на публичный архив с данными через Hydra:

```bash
uv run python -m shark_species_classifier.commands command=train data.yandex_public_url="https://disk.yandex.ru/d/<id>"
```

---

### Train

1. (Опционально) поднять MLflow server. По умолчанию проект ожидает `http://127.0.0.1:8080`:

```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

2. Запуск обучения:

```bash
uv run --active python -m shark_species_classifier.commands command=train `
  data.yandex_public_url='https://disk.360.yandex.ru/d/AvVGI04GbHC2Xw' `
  trainer.max_epochs=10
```
---

### Infer

Инференс по одному изображению (чекпойнт можно не указывать — будет взят самый новый из `checkpoints/`):

```bash
uv run python -m shark_species_classifier.commands command=infer infer.image_path=path/to/image.jpg
```

Пример с явным указанием чекпойнта:

```bash
uv run python -m shark_species_classifier.commands command=infer infer.image_path=path/to/image.jpg infer.checkpoint_path=checkpoints/some.ckpt
```
