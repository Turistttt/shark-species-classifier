### Shark species classification (13 classes)

Учебный MLOps‑проект: система компьютерного зрения, которая по фото определяет **вид
акулы** (13 классов) и отдаёт **вектор вероятностей softmax**.

#### Постановка задачи

- **Вход**: RGB‑изображение (JPG/PNG), приводимое к \(3 \times 224 \times 224\)
- **Выход**: вероятности по 13 видам (softmax)
- **Метрики**:
  - **Accuracy** (цель \(\ge 0.85\))
  - **Macro F1** (важно для редких видов)
- **Валидация/тест**:
  - фиксируем `random_state=2025`
  - делаем split **train/test = 80/20**
  - из train выделяем **val** (по умолчанию 10% от train), чтобы корректно выбирать
    лучший чекпойнт по `val_macro_f1` и делать early stopping

---

### Setup (uv)

Требования: Python 3.10+ и установленный `uv`.

1. Создать окружение:

```bash
uv venv
```

2. Установить зависимости:

```bash
uv sync --extra dev
```

3. Установить pre-commit хуки:

```bash
uv run pre-commit install
```

4. Проверить качество кода:

```bash
uv run pre-commit run -a
```

---

### Data (DVC + download)

По умолчанию проект использует данные в `sharks/` (папки‑классы, внутри
изображения). **Данные не должны храниться в git**, используйте DVC.

Если вы храните архив на Yandex Disk:

- положите публичную ссылку в `data.yandex_public_url` (см. `configs/data/default.yaml`)
- при запуске train/infer код попробует:
  1. `dvc pull`
  2. если не получилось — скачать zip по публичной ссылке и распаковать в `data.raw_dir`

---

### Train

Запуск тренировки:

```bash
uv run python -m shark_species_classifier.commands command=train
```

Пример: включить ResNet34 и задать число эпох:

```bash
uv run python -m shark_species_classifier.commands command=train model=resnet34 trainer.max_epochs=10
```

Логирование идёт в MLflow (по умолчанию `http://127.0.0.1:8080`) и содержит:

- метрики/лоссы (>= 3 графиков)
- гиперпараметры
- git commit id

---

### Infer

Инференс по одному изображению:

```bash
uv run python -m shark_species_classifier.commands \
  command=infer \
  infer.image_path=path/to/image.jpg \
  infer.checkpoint_path=path/to/best.ckpt
```
