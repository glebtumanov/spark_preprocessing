## Входы

### train_df (Spark)
Обучающая выборка, передаётся в `PreprocessingPipeline.run_preprocessing` как `train_sdf`.

### valid_df (Spark)
Валидационная выборка (опционально), передаётся как `valid_sdf`; используется вместе с train для шагов и дедупликации.

### scoring_df (Spark) опц.
Скоринговая выборка (опционально), проходит те же трансформации; не участвует в adversarial-отборе.

## Инициализация

### Hive context, Spark context
Используются для операций со Spark (`sqlContext`, `sc`) и сохранения в Hive при выборе формата `hive`.

### config.yaml
YAML-конфиг (см. `my_config.yaml`): пути артефактов, список `pipeline_steps`, схемы заполнения NA, параметры ребалансировки, кодирования, масштабирования и селекторов.

### PreprocessingPipeline()
Читает конфиг, настраивает артефакты и шаги, определяет служебные списки признаков и держатели `encoder`/`scaler`.

## Запуск

### .run_preprocessing()
Принимает Spark DataFrame'ы, опционально сэмплирует, инициализирует списки колонок, выполняет шаги из `pipeline_steps` по порядку, затем сохраняет артефакты и результаты.

## Базовые шаги

### sampling – сэмплирование данных
Случайная подвыборка `train_sdf`/`valid_sdf` по доле из конфига при `use_sampling=true`.

### определение типов фичей
Определяет `cat_cols`/`num_cols` по Spark-типам и `force_categorical`; исключает `index_col`, `target_col`, `skip_cols`.

### rebalancing – ребалансировка таргета
Даунсемплинг отрицательного класса до заданной доли положительного (`positive_frac`) в `train_sdf`.

### drop_forced_cols – удаление фичей по списку
Удаляет `force_drop_cols` из всех датасетов; пишет список в `artifacts/drop_forced_cols.txt` и обновляет списки признаков.

### convert_decimal – decimal → float
Кастует колонки `DecimalType` в `FloatType` (кроме `skip_cols`) во всех датасетах.

### fill_missing – заполнение пропусков
Заполняет NA по словарю из конфига: отдельные значения для числовых и категориальных колонок.

### drop_uninformative – удаление малоинформативных фичей
Удаляет признаки, где доля модального значения ≥ порога; сохраняет список и обновляет финальные списки.

### deduplicate – удаление дубликатов
Объединяет train+valid, удаляет дубли по `subset` (по умолчанию `index_col`), потом разделяет обратно; считает число удалённых строк.

## Преобразования признаков

### categorical_encoding – кодирование категориальных фичей
`label_encoder` (OrdinalEncoder) или `target_encoder`; фит на train-сэмпле, применение через `mapInPandas`; сохраняет `encoder.pkl`; при target encoding колонки становятся числовыми.

### scaling – масштабирование фичей
`standard`/`minmax`/`quantile`/`binning`; фит на train-сэмпле, применение через `mapInPandas` к `num_cols`; сохраняет `scaler.pkl`.

## Отбор признаков (Feature selectors)

### drop_correlated – удаление коррелирующих фичей (опц.)
Через `CorrFeatureRemover`: считает важности (или SHAP) на фолдах и для всех возможных пар удаляет одну из высоко коррелирующих фичей по меньшей важности; сохраняет список удалённых.

### drop_adversarial – удаление adversarial фичей (опц.)
Через `AdversarialFeatureRemover`: объединяет train/valid, учит модель отличать их; итеративно удаляет самые «важные» признаки до целевой метрики/лимитов; сохраняет историю и список удалённых.

### Forward Selection
Через `ForwardFeatureSelector`: считает важности, далее по одной добавляет фичи начиная с большей важности, принимая только улучшающие (или по «голосам» фолдов); исключает непринятые.

### Noise Selection
Через `NoiseFeatureSelector`: добавляет `noise_feature`, оставляет признаки, которые чаще шума по важности в доле фолдов ≥ порога; исключает остальные.

### Permutation Selection
Через `PermutationSelector`: считает permutation importance по фолдам; оставляет признаки выше порога в достаточном числе фолдов; исключает остальные.

## Артефакты и результаты

### save_artifacts_info – сохранение артефактов
Пишет `artifacts_info.json` с путями: списки удалённых/исключённых колонок, `encoder.pkl`, `scaler.pkl`.

### encoder.pkl / scaler.pkl
Сохранённые обученные кодировщик и скейлер для повторного применения.

### final_feature_list.txt / final_categorical_list.txt
Актуальные списки оставшихся признаков (общий и категориальные) после каждого удаления.

### save_results – сохранение выборок
Сохраняет `train_preprocessed`, `valid_preprocessed`, `scoring_preprocessed` в Parquet или Hive (по конфигу) с заданным суффиксом.


