# Обучение, оценка качества и артефакты (AirMouse)

## Аннотация

Документ описывает процедуру обучения модели классификации жестов в PyTorch, вычисление метрик качества и формирование артефактов (графики обучения, матрица ошибок, текстовый отчёт). Также фиксируются входные данные обучения (стратифицированное разбиение) и обеспечивается воспроизводимость.

## 1. Входные данные для обучения

Обучение производится по воспроизводимому разбиению:

- `data/splits/train.csv`
- `data/splits/test.csv`

Разбиение строится со стратификацией по `class_id`:

```bash
python scripts/split_dataset.py --test-size 0.2 --seed 42
```

## 2. Запуск обучения

Основной скрипт обучения: `airmouse/ml/train.py`.

Пример запуска:

```bash
python -m airmouse.ml.train --epochs 100 --lr 0.0005 --batch-size 64 --device cpu --log-dir logs/train
```

## 3. Что происходит в процессе обучения

В каждом эпохальном цикле:

- выполняется оптимизация функции потерь `CrossEntropyLoss` на train-части
- после каждой эпохи выполняется оценка на test-части
- сохраняется история метрик для построения графиков

## 4. Метрики качества

Вычисляются следующие показатели:

- **Accuracy** на тестовой выборке
- **Classification report**: precision/recall/F1 по каждому классу + macro/weighted средние
- **Confusion matrix**

## 5. Формируемые артефакты (визуализация процесса обучения)

При указании `--log-dir` сохраняются:

- [`training_curves.png`](../logs/experiments/run_030/training_curves.png) — кривые loss и accuracy по эпохам (пример: run_030)
- [`confusion_matrix.png`](../logs/experiments/run_030/confusion_matrix.png) — визуализация матрицы ошибок (пример: run_030)
- [`classification_report.txt`](../logs/experiments/run_030/classification_report.txt) — текстовый отчёт по метрикам (пример: run_030)

Пример вставки артефактов из одного из выполненных запусков (run_030):

![Кривые обучения (пример)](../logs/experiments/run_030/training_curves.png)

![Матрица ошибок (пример)](../logs/experiments/run_030/confusion_matrix.png)

Текстовый отчёт (пример, [`../logs/experiments/run_030/classification_report.txt`](../logs/experiments/run_030/classification_report.txt)):

```text
              precision    recall  f1-score   support

           0     0.9138    0.8833    0.8983        60
           1     0.9661    0.9500    0.9580        60
           2     0.9831    0.9667    0.9748        60
           3     0.9231    1.0000    0.9600        60
           4     1.0000    0.9836    0.9917        61

    accuracy                         0.9568       301
   macro avg     0.9572    0.9567    0.9566       301
weighted avg     0.9573    0.9568    0.9567       301
```

## 6. Экспорт обученной модели (прикладные “вытекающие”)

После обучения формируются:

- веса PyTorch: `models/gesture_model.pth`
- ONNX: `models/gesture_model.onnx`
- OpenVINO IR: `models/gesture_model.xml` (и сопутствующие файлы)

Это позволяет использовать модель в приложении в режиме инференса и сравнивать скорость/точность в разных движках.

## 7. Контроль воспроизводимости

Для воспроизводимых результатов рекомендуется:

- фиксировать `seed` при разбиении (`split_dataset.py`)
- фиксировать набор гиперпараметров (epochs, lr, batch_size)
- сохранять артефакты обучения в отдельный каталог

## Приложение A. Ссылки

- Краткая инструкция обучения: `reports/TRAINING.md`

