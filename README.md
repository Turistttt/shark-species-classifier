## shark-species-classifier

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
- **DVC**: трекинг данных (данные не коммитятся в git)

## Setup

Требования: установленный [`uv`](https://github.com/astral-sh/uv) и Git.
Перейдя в папку shark-species-classifier запускаем следующие команды:

```bash
uv venv
uv sync --extra dev

uv run pre-commit install
uv run pre-commit run -a
```

## Data

Данные **не хранятся в git**.

По умолчанию данные подтягиваются через **DVC remote** (Yandex Object Storage).
При запуске `train`/`infer` код сначала попробует сделать `dvc pull` через Python API.

Если папки `sharks/` нет и `dvc pull` не сработал, код скачает **zip‑архив** с данными по публичной ссылке Яндекс.Диска и распакует его в `raw_dir`.

Публичная ссылка на данные:

- `https://disk.360.yandex.ru/d/AvVGI04GbHC2Xw`

Ссылка задаётся через `data.yandex_public_url` (можно в `configs/data/default.yaml`, либо через override при запуске).

## Train

Опционально: поднять MLflow server (по умолчанию проект ожидает `http://127.0.0.1:8080`):

```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

Запуск обучения с 10ю эпохами (можно и меньше если не очень хочется ждать):

```powershell
uv run --active python -m shark_species_classifier.commands command=train `
  data.yandex_public_url='https://disk.360.yandex.ru/d/AvVGI04GbHC2Xw' `
  trainer.max_epochs=10
```

Результат на 10ти эпохах:

<img width="623" height="133" alt="image" src="https://github.com/user-attachments/assets/bdf93c43-a427-4369-ad90-f96542cc5a83" />

Примеры override параметров:

```powershell
uv run --active python -m shark_species_classifier.commands command=train trainer.max_epochs=3
uv run --active python -m shark_species_classifier.commands command=train model=cnn data.batch_size=64
```

## Infer

Инференс по одному изображению (если `infer.checkpoint_path` не задан, будет взят самый новый чекпойнт из `checkpoints/`):

```powershell
uv run --active python -m shark_species_classifier.commands command=infer `
  infer.image_path=path/to/image.jpg
```

Пример инференса :
<img width="977" height="122" alt="image" src="https://github.com/user-attachments/assets/a8726bf3-0c64-411b-ae69-727ef1acdea0" />
