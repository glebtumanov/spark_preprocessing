# Spark Feature Preprocessing Pipeline

## Назначение

Гибкая и масштабируемая система для автоматизированной предобработки и отбора признаков в задачах машинного обучения на больших данных с использованием PySpark и LightGBM.

## Возможности
- Очистка, обработка пропусков, преобразование типов, масштабирование, кодирование категориальных признаков
- Балансировка классов, удаление дубликатов, удаление неинформативных и коррелированных признаков
- Современные методы отбора признаков: корреляция, adversarial selection, noise selection, forward selection
- Гибкая настройка через YAML-конфиг
- Сохранение артефактов и результатов в parquet/hive
- Расширяемая архитектура (базовые абстракции для селекторов признаков)

## Требования
- Python 3.12+
- PySpark >= 3.2
- LightGBM
- pandas, numpy, scikit-learn, category_encoders, pyyaml

## Пример запуска
```bash
python start_preprocessing.py my_config.yaml
```

## Структура проекта
- `preprocessing.py` — основной пайплайн предобработки
- `feature_selector.py` — базовый интерфейс селекторов признаков
- `corr_feature_remover.py`, `adversarial_feature_remover.py`, `noise_feature_selector.py`, `forward_selection.py` — реализации методов отбора признаков
- `my_config.yaml` — пример конфига
- `start_preprocessing.py` — точка входа для запуска пайплайна

## Контакты
Вопросы и предложения: @your_team