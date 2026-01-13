# AirMouse

Проект **AirMouse** — система управления курсором с помощью жестов кисти, распознаваемых по 3D-координатам landmarks (MediaPipe Hands). Репозиторий включает:

- desktop-приложение (GUI) для использования жестов на ПК;
- пайплайн формирования датасета жестов;
- обучение модели классификации жестов (PyTorch) с оценкой качества и сохранением артефактов;
- экспорт модели в **ONNX** и **OpenVINO IR** для ускорения инференса.

## Документация

- Описание структуры датасета: [`data/DESCRIPTION.md`](./data/DESCRIPTION.md)
- Формирование и разметка датасета: [`docs/01_dataset_creation_and_labeling.md`](./docs/01_dataset_creation_and_labeling.md)
- Обучение и оценка качества: [`docs/02_training_and_evaluation.md`](./docs/02_training_and_evaluation.md)
- Протокол экспериментов: [`docs/03_experiments_protocol.md`](./docs/03_experiments_protocol.md)
- Гайд по приложению [`docs/04_app_usage_guide.md`](./docs/04_app_usage_guide.md)

