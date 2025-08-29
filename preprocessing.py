#!/bin/env python
import os
import json
import yaml
import pickle
import shutil
import numpy as np
import pandas as pd
import multiprocessing
import glob
import tempfile

import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
    KBinsDiscretizer
)
from sklearn.metrics import roc_auc_score, mean_squared_error
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.target_encoder import TargetEncoder

from pyspark.sql import DataFrame # type: ignore
from pyspark.sql import functions as F # type: ignore
from pyspark.sql import types as T # type: ignore
from pyspark.sql.functions import pandas_udf, col # type: ignore

from feature_selectors import (
    BaseFeatureSelector,
    CorrFeatureRemover,
    AdversarialFeatureRemover,
    NoiseFeatureSelector,
    ForwardFeatureSelector,
    PermutationSelector
)

sklearn.set_config(transform_output='pandas')

# ===== Вспомогательные функции для работы с файлами и данными =====

def ensure_dir_exists(dir_path):
    """Создает директорию, если она не существует"""
    os.makedirs(dir_path, exist_ok=True)


def save_as_parquet(df, output_file_path):
    """Сохраняет DataFrame в parquet файл"""
    ensure_dir_exists(os.path.dirname(output_file_path))

    # Создаем временную директорию
    temp_dir = tempfile.mkdtemp()

    try:
        # Сохраняем df в одну партицию
        df.coalesce(1).write.mode("overwrite").parquet(temp_dir)

        # Ищем сгенерированный parquet-файл
        part_files = glob.glob(os.path.join(temp_dir, "part-*.parquet"))
        if not part_files:
            raise FileNotFoundError("Не найден файл part-*.parquet в временной директории.")

        # Перемещаем файл
        shutil.move(part_files[0], output_file_path)
    finally:
        # Удаляем временную директорию
        shutil.rmtree(temp_dir)


def unpivot(df: DataFrame, cols, name_col: str, value_col: str) -> DataFrame:
    """Преобразует DataFrame из широкого формата в длинный"""
    unpvt = F.explode(
        F.array(*[
            F.struct(F.lit(c).alias('col_name'), F.col(c).alias('col_value'))
            for c in cols
        ])
    ).alias('unpvt')

    unchanged_cols = [c for c in df.columns if c not in cols]

    return (
        df
        .select(*unchanged_cols, unpvt)
        .select(
            *unchanged_cols,
            F.col('unpvt.col_name').alias(name_col),
            F.col('unpvt.col_value').alias(value_col)
        )
    )


def save_columns(col_list, path):
    """Сохраняет список колонок в файл"""
    ensure_dir_exists(os.path.dirname(path))
    with open(path, "w") as f:
        for c in col_list:
            f.write(str(c) + "\n")


def load_columns(path):
    """Загружает список колонок из файла"""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


# ===== UDF функции для Spark =====

def get_ordinal_encoder_udf(col_names, encoder):
    """UDF для применения ordinal encoding в Spark"""
    def ordinal_encoder_pdf(iterator):
        for pdf in iterator:
            pdf[col_names] = encoder.transform(pdf[col_names]).astype(np.int32)
            yield pdf
    return ordinal_encoder_pdf


def get_target_encoder_udf(col_names, encoder):
    """UDF для применения target encoding в Spark"""
    def target_encoder_pdf(iterator):
        for pdf in iterator:
            pdf[col_names] = encoder.transform(pdf[col_names]).astype(np.float32)
            yield pdf
    return target_encoder_pdf


def get_transform_udf(col_names, scaler):
    """UDF для масштабирования в Spark"""
    def transform_pdf(iterator):
        for pdf in iterator:
            pdf[col_names] = scaler.transform(pdf[col_names]).astype(np.float32)
            yield pdf
    return transform_pdf


class PreprocessingPipeline:
    """
    Класс для предобработки данных на Spark DataFrame.
    Одновременно обрабатывает train, test и scoring датасеты.
    """

    def __init__(self, config_path, sc, sqlContext):
        """
        Инициализация пайплайна предобработки.

        Args:
            config_path: путь к yaml-файлу с конфигурацией
            sc: SparkContext
            sqlContext: SparkSession
        """
        self.sc = sc
        self.sqlContext = sqlContext

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Директория для артефактов
        self.artifacts_path = self.config["artifacts"]["path"]
        self.clear_artifacts = bool(self.config["artifacts"]["clear"])
        ensure_dir_exists(self.artifacts_path)

        self.sampling = self.config["sampling"]
        self.train_sample_size = self.sampling.get('train_sample_size', None)
        self.test_sample_size = self.sampling.get('test_sample_size', None)
        self.use_sampling = self.sampling.get('use_sampling', False)

        # Основные настройки
        self.index_col = self.config["common"]["index_col"]
        self.target_col = self.config["common"].get("target_col")
        self.task_type = self.config["common"].get("task_type", "binary_classification")
        self.skip_cols = set(self.config["common"].get("skip_cols", []))
        self.force_drop_cols = set(self.config["common"].get("force_drop_cols", []))

        # Настройка параллелизма
        self.n_jobs = self.config["common"].get("n_jobs", -1)
        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()

        # Список шагов пайплайна
        self.pipeline_steps = self.config.get("pipeline_steps", [])

        # Трансформеры
        self.encoder = None
        self.scaler = None
        # Флаг: выполнено ли кодирование категориальных признаков
        self.is_categorical_encoded = False

        # Списки колонок для удаления
        self.uninformative_cols = []
        self.correlated_cols = []
        self.adversarial_cols = []
        self.noise_cols = []
        self.fs_cols = []

        # Текущие списки колонок
        self.cat_cols = []
        self.num_cols = []
        self.final_features_list = []

        # Пути к артефактам
        self.artifacts_info = {
            "drop_forced_cols_path": None,
            "drop_uninformative_cols_path": None,
            "drop_correlated_cols_path": None,
            "drop_adversarial_cols_path": None,
            "noise_excluded_cols_path": None,
            "fs_excluded_cols_path": None,
            "permutation_excluded_cols_path": None,
            "encoder_path": None,
            "scaler_path": None
        }

        # Пути для финальных списков признаков
        self.path_features = os.path.join(self.artifacts_path, "final_feature_list.txt")
        self.path_categorical = os.path.join(self.artifacts_path, "final_categorical_list.txt")

        # DataFrame для обучения/тестирования/скоринга
        self.train_sdf = None
        self.test_sdf = None
        self.scoring_sdf = None

    # ============ Основной процесс обработки ============

    def run_preprocessing(self, train_sdf, valid_sdf=None, scoring_sdf=None):
        """
        Полный цикл предобработки данных с использованием предоставленных DataFrame

        Args:
            train_sdf: Spark DataFrame с обучающими данными
            valid_sdf: Spark DataFrame с валидационными данными (опционально)
            scoring_sdf: Spark DataFrame с данными для скоринга (опционально)

        Returns:
            Кортеж (train_sdf, valid_sdf, scoring_sdf) с обработанными данными
        """
        # Сохраняем переданные DataFrame
        self.train_sdf = train_sdf
        self.test_sdf = valid_sdf
        self.scoring_sdf = scoring_sdf

        # Сэмплируем данные
        if self.use_sampling:
            if self.train_sample_size:
                fraction = 1.0
                cnt = self.train_sdf.count()
                if self.train_sample_size and int(self.train_sample_size) < cnt:
                    fraction = self.train_sample_size / float(cnt)
                self.train_sdf = self.train_sdf.sample(withReplacement=False, fraction=fraction, seed=42)
                print(f"[Sampling] train size: {self.train_sdf.count()}")
            if self.test_sample_size:
                fraction = 1.0
                cnt = self.test_sdf.count()
                if self.test_sample_size and int(self.test_sample_size) < cnt:
                    fraction = self.test_sample_size / float(cnt)
                self.test_sdf = self.test_sdf.sample(withReplacement=False, fraction=fraction, seed=42)
                print(f"[Sampling] test size: {self.test_sdf.count()}")

        # Инициализация списков колонок
        self._init_column_lists()

        # Очистка артефактов при необходимости
        if self.clear_artifacts:
            self._clear_artifacts_folder()

        # Ребалансировка, если требуется
        if "rebalancing" in self.pipeline_steps:
            self.rebalancing_train()

        # Выполнение шагов пайплайна
        for step in self.pipeline_steps:
            if step == "rebalancing":
                continue  # Уже обработано
            elif step in self.preprocessing_methods:
                print(f"Start processing step: {step}")
                method = self.preprocessing_methods[step]
                method(self)
            else:
                print(f"[WARN] Неизвестный шаг пайплайна: {step}")

        # Сохранение информации и результатов
        self.save_artifacts_info()
        self.save_results()

        return self.train_sdf, self.test_sdf, self.scoring_sdf

    # ============ Методы баланcировки ============

    def rebalancing_train(self):
        """Ребалансировка данных обучающей выборки"""
        if self.train_sdf is None:
            print("[WARN] train_sdf не задан, ребалансировка невозможна")
            return

        positive_frac = self.config["preprocessing"]["rebalancing"].get("positive_frac", 1.0)
        print(f"[rebalancing] new positive frac: {positive_frac}")

        # Рассчитываем коэффициент для ребалансировки
        pos_count = self.train_sdf.filter(F.col(self.target_col) == 1).count()
        neg_count = self.train_sdf.filter(F.col(self.target_col) == 0).count()
        neg_frac = pos_count * (1 / positive_frac) / neg_count

        # Применяем ребалансировку
        self.train_sdf = self.train_sdf.filter(
            (F.col(self.target_col) == 1) | (F.rand() <= neg_frac)
        )

        # Отображаем распределение целевой переменной
        self.train_sdf.groupBy(self.target_col).agg(F.count(F.lit(1)).alias('cnt')).show()
        print(f"[rebalancing] Ребалансировка завершена, итого строк: {self.train_sdf.count()}")

    # ============ Методы предобработки данных ============

    def drop_forced_cols(self):
        """Удаление принудительно исключаемых колонок"""
        self._check_data_loaded()

        # Получаем список колонок для принудительного удаления
        cols_to_drop = list(self.force_drop_cols)

        if cols_to_drop:
            print(f"[dropForcedCols] Удаляем колонки: {len(cols_to_drop)}")

            # Удаляем колонки из всех датасетов
            if self.train_sdf is not None:
                self.train_sdf = self.train_sdf.drop(*cols_to_drop)
            if self.test_sdf is not None:
                self.test_sdf = self.test_sdf.drop(*cols_to_drop)
            if self.scoring_sdf is not None:
                self.scoring_sdf = self.scoring_sdf.drop(*cols_to_drop)

            # Сохраняем список удаленных колонок
            path = os.path.join(self.artifacts_path, "drop_forced_cols.txt")
            save_columns(cols_to_drop, path)
            self.artifacts_info["drop_forced_cols_path"] = path

            # Обновляем списки колонок
            self._remove_cols_from_lists(self.force_drop_cols)

    def convert_decimal(self):
        """Преобразование колонок с типом Decimal в Float"""
        self._check_data_loaded()

        # Функция для преобразования Decimal в Float для одного датасета
        def convert_decimal_in_df(df):
            if df is None:
                return None

            for field in df.schema.fields:
                if field.name in self.skip_cols:
                    continue
                if isinstance(field.dataType, T.DecimalType):
                    df = df.withColumn(
                        field.name,
                        F.col(field.name).cast(T.FloatType())
                    )
            return df

        # Применяем преобразование ко всем датасетам
        self.train_sdf = convert_decimal_in_df(self.train_sdf)
        self.test_sdf = convert_decimal_in_df(self.test_sdf)
        self.scoring_sdf = convert_decimal_in_df(self.scoring_sdf)

    def fill_missing(self):
        """Заполнение пропущенных значений"""
        self._check_data_loaded()

        # Получаем значения для заполнения пропусков
        numeric_na_val = self.config["preprocessing"]["missing_values"]["numeric"]
        categorical_na_val = self.config["preprocessing"]["missing_values"]["categorical"]

        # Формируем словарь для заполнения
        fill_dict = {}
        for c in self.cat_cols:
            if c not in self.skip_cols:
                fill_dict[c] = categorical_na_val
        for c in self.num_cols:
            if c not in self.skip_cols:
                fill_dict[c] = numeric_na_val

        # Применяем заполнение ко всем датасетам
        if fill_dict:
            if self.train_sdf is not None:
                self.train_sdf = self.train_sdf.fillna(fill_dict)
            if self.test_sdf is not None:
                self.test_sdf = self.test_sdf.fillna(fill_dict)
            if self.scoring_sdf is not None:
                self.scoring_sdf = self.scoring_sdf.fillna(fill_dict)

    def drop_uninformative(self):
        """Удаление колонок с низкой информативностью"""
        self._check_data_loaded()

        # Получаем порог для удаления
        threshold = self.config["preprocessing"]["drop_uninformative"]["threshold"]

        # Определяем колонки для проверки
        skip_cols_local = {self.index_col} | self.skip_cols
        if self.target_col:
            skip_cols_local.add(self.target_col)

        columns_to_check = [c for c in self.train_sdf.columns if c not in skip_cols_local]
        if not columns_to_check:
            return []

        # Анализируем распределение значений
        long_df = unpivot(self.train_sdf, columns_to_check, 'feature_name', 'feature_value')
        grouped = long_df.groupBy('feature_name', 'feature_value').agg(F.count('*').alias('cnt'))

        stats = (grouped
                .groupBy('feature_name')
                .agg(F.sum('cnt').alias('total_cnt'), F.max('cnt').alias('max_cnt'))
                .withColumn('ratio', F.col('max_cnt') / F.col('total_cnt'))
                .filter(F.col('ratio') >= threshold)
              )

        # Получаем колонки для удаления
        found_cols = [row['feature_name'] for row in stats.select('feature_name').collect()]
        found_cols = [c for c in found_cols if c not in self.skip_cols]

        if found_cols:
            print(f"[dropUninformative] Удаляем колонки (>{threshold*100}% одного значения): {len(found_cols)}")

            # Удаляем колонки из всех датасетов
            if self.train_sdf is not None:
                self.train_sdf = self.train_sdf.drop(*found_cols)
            if self.test_sdf is not None:
                self.test_sdf = self.test_sdf.drop(*found_cols)
            if self.scoring_sdf is not None:
                self.scoring_sdf = self.scoring_sdf.drop(*found_cols)

            # Сохраняем список удаленных колонок
            path = os.path.join(self.artifacts_path, "drop_uninformative_cols.txt")
            save_columns(found_cols, path)
            self.artifacts_info["drop_uninformative_cols_path"] = path

            # Обновляем списки колонок
            self._remove_cols_from_lists(found_cols)

    def deduplicate(self):
        """Удаление дубликатов"""
        subset = self.config["preprocessing"]["deduplicate"].get("subset", [self.index_col])
        print(f"[deduplicate] subset: [{', '.join(subset)}]")

        self._check_data_loaded()

        # Обработка train и test датасетов вместе (если есть test)
        if self.test_sdf is not None:
            # Помечаем строки для идентификации источника
            self.train_sdf = self.train_sdf.withColumn('is_train_', F.lit(1))
            self.test_sdf = self.test_sdf.withColumn('is_train_', F.lit(0))

            combined_sdf = self.train_sdf.unionByName(self.test_sdf)
            initial_count = combined_sdf.count()

            # Удаляем дубликаты
            unique_sdf = combined_sdf.dropDuplicates(subset=subset)
            final_count = unique_sdf.count()
            removed = initial_count - final_count

            # Разделяем обратно на train и test
            self.train_sdf = unique_sdf.filter(F.col('is_train_')==1).drop('is_train_')
            self.test_sdf = unique_sdf.filter(F.col('is_train_')==0).drop('is_train_')
            print(f"[deduplicate] Удалено {removed} строк из train и test")

        else:
            # Дедупликация только для train
            initial_count = self.train_sdf.count()
            self.train_sdf = self.train_sdf.dropDuplicates(subset=subset)
            final_count = self.train_sdf.count()
            removed = initial_count - final_count
            print(f"[deduplicate] Удалено {removed} строк из train")

        # Дедупликация scoring датасета (отдельно)
        # if self.scoring_sdf is not None:
        #     initial_count = self.scoring_sdf.count()
        #     self.scoring_sdf = self.scoring_sdf.dropDuplicates(subset=subset)
        #     final_count = self.scoring_sdf.count()
        #     removed = initial_count - final_count
        #     print(f"[deduplicate] Удалено {removed} строк из scoring")

    def categorical_encoding(self):
        """Кодирование категориальных признаков"""
        self._check_data_loaded()

        method = self.config["preprocessing"]["categorical_encoding"]["method"]
        cat_cols = list(self.cat_cols)

        if not cat_cols:
            return

        encoder_path = os.path.join(self.artifacts_path, "encoder.pkl")

        # Обучаем энкодер на train датасете
        sample_size = self.config["preprocessing"]["categorical_encoding"].get("sample_size")

        if sample_size:
            train_sample_pdf = self._sample_for_fit(self.train_sdf, cat_cols, sample_size=sample_size)
        else:
            train_sample_pdf = self.train_sdf.toPandas()

        if method == "label_encoder":
            self.encoder = OrdinalEncoder(cols=cat_cols, drop_invariant=True, handle_unknown='value')
            self.encoder.fit(train_sample_pdf[cat_cols])
        elif method == "target_encoder":
            self.encoder = TargetEncoder(min_samples_leaf=100, smoothing=20.0)
            self.encoder.fit(train_sample_pdf[cat_cols], train_sample_pdf[self.target_col])

            # При target_encoding считаем фичи числовыми
            for c in cat_cols:
                if c in self.cat_cols:
                    self.cat_cols.remove(c)
                if c not in self.num_cols:
                    self.num_cols.append(c)
        else:
            raise ValueError(f"Неизвестный метод кодирования: {method}")

        # Сохраняем энкодер
        ensure_dir_exists(os.path.dirname(encoder_path))
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.encoder, f)
        self.artifacts_info["encoder_path"] = encoder_path

        # Подготовка UDF для трансформации
        if method == "label_encoder":
            fields = [T.StructField(f.name, T.IntegerType(), True) if f.name in cat_cols else f
                     for f in self.train_sdf.schema.fields]
            schema = T.StructType(fields)
            func = get_ordinal_encoder_udf(cat_cols, encoder=self.encoder)
        elif method == "target_encoder":
            fields = [T.StructField(f.name, T.FloatType(), True) if f.name in cat_cols else f
                     for f in self.train_sdf.schema.fields]
            schema = T.StructType(fields)
            func = get_target_encoder_udf(cat_cols, encoder=self.encoder)

        # Применяем трансформацию ко всем датасетам
        if self.train_sdf is not None:
            self.train_sdf = self.train_sdf.mapInPandas(func, schema=schema)
        if self.test_sdf is not None:
            # Для test используем схему из train
            test_fields = [T.StructField(f.name,
                                         next((field.dataType for field in fields if field.name == f.name), f.dataType),
                                         True)
                           for f in self.test_sdf.schema.fields]
            test_schema = T.StructType(test_fields)
            self.test_sdf = self.test_sdf.mapInPandas(func, schema=test_schema)
        if self.scoring_sdf is not None:
            # Для scoring используем схему из train
            scoring_fields = [T.StructField(f.name,
                                           next((field.dataType for field in fields if field.name == f.name), f.dataType),
                                           True)
                              for f in self.scoring_sdf.schema.fields]
            scoring_schema = T.StructType(scoring_fields)
            self.scoring_sdf = self.scoring_sdf.mapInPandas(func, schema=scoring_schema)
        # Отмечаем, что кодирование категориальных признаков выполнено
        self.is_categorical_encoded = True

        # После изменения списков признаков/типов фиксируем артефакты
        self._persist_feature_lists()

    def scaling(self):
        """Масштабирование числовых признаков"""
        self._check_data_loaded()

        num_cols = list(self.num_cols)
        if not num_cols:
            return

        scaler_path = os.path.join(self.artifacts_path, "scaler.pkl")

        # Обучаем скейлер на train датасете
        sample_size = self.config["preprocessing"]["scaling"].get("sample_size")

        if sample_size:
            train_sample_pdf = self._sample_for_fit(self.train_sdf, num_cols, sample_size=sample_size)
        else:
            train_sample_pdf = self.train_sdf.toPandas()

        method = self.config["preprocessing"]["scaling"]["method"]

        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "quantile":
            self.scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
        elif method == "binning":
            bins = self.config["preprocessing"]["scaling"].get("bins", 10)
            self.scaler = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
        else:
            raise ValueError(f"Неизвестный метод масштабирования: {method}")

        self.scaler.fit(train_sample_pdf[num_cols])

        # Сохраняем скейлер
        ensure_dir_exists(os.path.dirname(scaler_path))
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        self.artifacts_info["scaler_path"] = scaler_path

        # Подготовка UDF для трансформации
        transform_func = get_transform_udf(num_cols, scaler=self.scaler)

        # Применяем трансформацию ко всем датасетам
        if self.train_sdf is not None:
            fields = [T.StructField(f.name, T.FloatType(), True) if f.name in num_cols else f
                     for f in self.train_sdf.schema.fields]
            schema = T.StructType(fields)
            self.train_sdf = self.train_sdf.mapInPandas(transform_func, schema=schema)

        if self.test_sdf is not None:
            test_fields = [T.StructField(f.name, T.FloatType(), True) if f.name in num_cols else f
                          for f in self.test_sdf.schema.fields]
            test_schema = T.StructType(test_fields)
            self.test_sdf = self.test_sdf.mapInPandas(transform_func, schema=test_schema)

        if self.scoring_sdf is not None:
            scoring_fields = [T.StructField(f.name, T.FloatType(), True) if f.name in num_cols else f
                             for f in self.scoring_sdf.schema.fields]
            scoring_schema = T.StructType(scoring_fields)
            self.scoring_sdf = self.scoring_sdf.mapInPandas(transform_func, schema=scoring_schema)

    def drop_correlated(self):
        """Удаление сильно коррелирующих признаков"""
        self._check_data_loaded()
        selector = CorrFeatureRemover(self.config, self.sc)

        # Используем только числовые признаки для корреляционного анализа
        features = self._get_numeric_features_for_modeling()
        if not features:
            print("[drop_correlated] Нет числовых признаков для анализа, шаг пропущен")
            return

        # Получаем список признаков для исключения (категориальные не используются здесь)
        features_to_exclude = selector.get_features_to_exclude(
            train_sdf=self.train_sdf,
            features=features,
            categorical_features=[]
        )

        if features_to_exclude:
            print(f"[drop_correlated] Удаляем колонки: {len(features_to_exclude)}")

            # Удаляем колонки из всех датасетов
            if self.train_sdf is not None:
                self.train_sdf = self.train_sdf.drop(*features_to_exclude)
            if self.test_sdf is not None:
                self.test_sdf = self.test_sdf.drop(*features_to_exclude)
            if self.scoring_sdf is not None:
                self.scoring_sdf = self.scoring_sdf.drop(*features_to_exclude)

            # Сохраняем список исключенных признаков
            path = os.path.join(self.artifacts_path, "drop_correlated_cols.txt")
            save_columns(features_to_exclude, path)
            self.artifacts_info["drop_correlated_cols_path"] = path

            # Обновляем списки колонок
            self._remove_cols_from_lists(features_to_exclude)

    def drop_adversarial(self):
        """Удаление adversarial признаков - ВАЖНО: scoring_sdf не участвует в анализе"""
        self._check_data_loaded()
        selector = AdversarialFeatureRemover(self.config, self.sc)

        # Получаем список признаков для исключения (scoring не участвует)
        if self.is_categorical_encoded:
            features = self.final_features_list
            categorical_features = self.cat_cols
        else:
            features = self._get_numeric_features_for_modeling()
            categorical_features = []

        if not features:
            print("[drop_adversarial] Нет признаков для анализа, шаг пропущен")
            return

        features_to_exclude = selector.get_features_to_exclude(
            train_sdf=self.train_sdf,
            test_sdf=self.test_sdf,
            features=features,
            categorical_features=categorical_features
        )

        if features_to_exclude:
            print(f"[drop_adversarial] Удаляем колонки: {len(features_to_exclude)}")

            # Удаляем колонки из всех датасетов
            if self.train_sdf is not None:
                self.train_sdf = self.train_sdf.drop(*features_to_exclude)
            if self.test_sdf is not None:
                self.test_sdf = self.test_sdf.drop(*features_to_exclude)
            if self.scoring_sdf is not None:
                self.scoring_sdf = self.scoring_sdf.drop(*features_to_exclude)

            # Сохраняем список исключенных признаков
            path = os.path.join(self.artifacts_path, "drop_adversarial_cols.txt")
            save_columns(features_to_exclude, path)
            self.artifacts_info["drop_adversarial_cols_path"] = path

            # Обновляем списки колонок
            self._remove_cols_from_lists(features_to_exclude)

    def select_noise(self):
        """Отбор признаков на основе шума"""
        self._check_data_loaded()
        selector = NoiseFeatureSelector(self.config, self.sc)

        # Получаем список признаков для включения
        if self.is_categorical_encoded:
            features = self.final_features_list
            categorical_features = self.cat_cols
        else:
            features = self._get_numeric_features_for_modeling()
            categorical_features = []

        if not features:
            print("[select_noise] Нет признаков для анализа, шаг пропущен")
            return

        features_to_include = selector.get_features_to_include(
            train_sdf=self.train_sdf,
            test_sdf=None,
            features=features,
            categorical_features=categorical_features
        )

        # Исключаем только из анализируемого множества features
        features_to_exclude = [col for col in self.final_features_list if (col in features and col not in features_to_include)]

        if features_to_exclude:
            print(f"[select_noise] Удаляем колонки: {len(features_to_exclude)}")

            # Удаляем колонки из всех датасетов
            if self.train_sdf is not None:
                self.train_sdf = self.train_sdf.drop(*features_to_exclude)
            if self.test_sdf is not None:
                self.test_sdf = self.test_sdf.drop(*features_to_exclude)
            if self.scoring_sdf is not None:
                self.scoring_sdf = self.scoring_sdf.drop(*features_to_exclude)

            # Сохраняем список исключенных признаков
            path = os.path.join(self.artifacts_path, "noise_excluded_cols.txt")
            save_columns(features_to_exclude, path)
            self.artifacts_info["noise_excluded_cols_path"] = path

            # Обновляем списки колонок
            self._remove_cols_from_lists(features_to_exclude)

    def select_forward(self):
        """Отбор признаков с помощью прямого отбора"""
        self._check_data_loaded()
        selector = ForwardFeatureSelector(self.config, self.sc)

        # Получаем список признаков для включения
        if self.is_categorical_encoded:
            features = self.final_features_list
            categorical_features = self.cat_cols
        else:
            features = self._get_numeric_features_for_modeling()
            categorical_features = []

        if not features:
            print("[select_forward] Нет признаков для анализа, шаг пропущен")
            return

        features_to_include = selector.get_features_to_include(
            train_sdf=self.train_sdf,
            test_sdf=None,
            features=features,
            categorical_features=categorical_features
        )

        # Исключаем только из анализируемого множества features
        features_to_exclude = [col for col in self.final_features_list if (col in features and col not in features_to_include)]

        if features_to_exclude:
            print(f"[select_forward] Удаляем колонки: {len(features_to_exclude)}")

            # Удаляем колонки из всех датасетов
            if self.train_sdf is not None:
                self.train_sdf = self.train_sdf.drop(*features_to_exclude)
            if self.test_sdf is not None:
                self.test_sdf = self.test_sdf.drop(*features_to_exclude)
            if self.scoring_sdf is not None:
                self.scoring_sdf = self.scoring_sdf.drop(*features_to_exclude)

            # Сохраняем список исключенных признаков
            path = os.path.join(self.artifacts_path, "fs_excluded_cols.txt")
            save_columns(features_to_exclude, path)
            self.artifacts_info["fs_excluded_cols_path"] = path

            # Обновляем списки колонок
            self._remove_cols_from_lists(features_to_exclude)

    def select_permutation(self):
        """Отбор признаков с помощью permutation importance"""
        self._check_data_loaded()
        selector = PermutationSelector(self.config, self.sc)

        # Получаем список признаков для включения
        if self.is_categorical_encoded:
            features = self.final_features_list
            categorical_features = self.cat_cols
        else:
            features = self._get_numeric_features_for_modeling()
            categorical_features = []

        if not features:
            print("[select_permutation] Нет признаков для анализа, шаг пропущен")
            return

        features_to_include = selector.get_features_to_include(
            train_sdf=self.train_sdf,
            test_sdf=None,
            features=features,
            categorical_features=categorical_features
        )

        # Исключаем только из анализируемого множества features
        features_to_exclude = [col for col in self.final_features_list if (col in features and col not in features_to_include)]

        if features_to_exclude:
            print(f"[select_permutation] Удаляем колонки: {len(features_to_exclude)}")

            # Удаляем колонки из всех датасетов
            if self.train_sdf is not None:
                self.train_sdf = self.train_sdf.drop(*features_to_exclude)
            if self.test_sdf is not None:
                self.test_sdf = self.test_sdf.drop(*features_to_exclude)
            if self.scoring_sdf is not None:
                self.scoring_sdf = self.scoring_sdf.drop(*features_to_exclude)

            # Сохраняем список исключенных признаков
            path = os.path.join(self.artifacts_path, "permutation_excluded_cols.txt")
            save_columns(features_to_exclude, path)
            self.artifacts_info["permutation_excluded_cols_path"] = path

            # Обновляем списки колонок
            self._remove_cols_from_lists(features_to_exclude)

    # ============ Вспомогательные методы ============

    def _init_column_lists(self):
        """Инициализация списков категориальных и числовых колонок"""
        if self.train_sdf is None:
            return

        forced_cat = set(self.config["preprocessing"]["categorical_encoding"].get("force_categorical", []))
        cat_cols_local = []
        num_cols_local = []
        all_cols_local = []

        # Проходим по всем полям и классифицируем их
        for field in self.train_sdf.schema.fields:
            col_name = field.name

            # Пропускаем индекс, таргет и колонки из skip_cols
            if col_name in [self.index_col, self.target_col] or col_name in self.skip_cols:
                continue

            # Добавляем в список категориальных, если задано явно
            if col_name in forced_cat:
                cat_cols_local.append(col_name)
                all_cols_local.append(col_name)
                continue

            # Определяем тип по схеме
            if isinstance(field.dataType, T.StringType):
                cat_cols_local.append(col_name)
                all_cols_local.append(col_name)
            elif isinstance(field.dataType, (T.IntegerType, T.LongType, T.FloatType,
                                         T.DoubleType, T.ShortType, T.BooleanType,
                                         T.ByteType, T.DecimalType)):
                num_cols_local.append(col_name)
                all_cols_local.append(col_name)
            else:
                print(f"[WARN] Неизвестный тип: {field.dataType} (колонка {col_name})")

        # Сохраняем списки
        self.cat_cols = cat_cols_local
        self.num_cols = num_cols_local
        self.final_features_list = all_cols_local

        # Сохраняем начальные списки, чтобы сформировать артефакты даже без шагов отбора/кодирования
        self._persist_feature_lists()

    def _remove_cols_from_lists(self, cols_to_remove):
        """Удаляет колонки из списков и сохраняет обновленные списки"""
        for c in cols_to_remove:
            if c in self.cat_cols:
                self.cat_cols.remove(c)
            if c in self.num_cols:
                self.num_cols.remove(c)
            if c in self.final_features_list:
                self.final_features_list.remove(c)
        # Сохраняем обновленные списки через общий метод
        self._persist_feature_lists()

    def _persist_feature_lists(self):
        """Сохраняет текущие финальные списки признаков и категориальных признаков в артефакты."""
        ensure_dir_exists(os.path.dirname(self.path_features))
        ensure_dir_exists(os.path.dirname(self.path_categorical))

        with open(self.path_features, 'w') as f:
            f.write('\n'.join(self.final_features_list))

        with open(self.path_categorical, 'w') as f:
            f.write('\n'.join(self.cat_cols))

        print(
            f"[Feature list] всего признаков: {len(self.final_features_list)}, категориальных: {len(self.cat_cols)}"
        )

    def _get_numeric_features_for_modeling(self):
        """Возвращает список числовых признаков из текущего финального пула."""
        return [c for c in self.final_features_list if c in self.num_cols]

    def _check_data_loaded(self):
        """Проверка, что данные загружены"""
        if self.train_sdf is None:
            raise ValueError("train_sdf не задан. Используйте метод run_preprocessing с указанием DataFrame.")

    def _sample_for_fit(self, sdf: DataFrame, cols, sample_size=10000):
        """Сэмплирование данных для обучения трансформеров"""
        cnt = sdf.count()
        if cnt == 0:
            return pd.DataFrame(columns=cols)

        # Определяем долю для сэмплирования
        fraction = 1.0
        if sample_size and int(sample_size) < cnt:
            fraction = sample_size / float(cnt)

        # Сэмплируем данные
        sdf_sample = sdf.sample(withReplacement=False, fraction=fraction, seed=42)

        # Выбираем нужные колонки
        if self.target_col in sdf.columns:
            sdf_sample = sdf_sample.select(*cols, self.target_col)
        else:
            sdf_sample = sdf_sample.select(*cols)

        # Преобразуем в pandas
        return sdf_sample.toPandas()

    def _clear_artifacts_folder(self):
        """Очистка папки артефактов"""
        for filename in os.listdir(self.artifacts_path):
            file_path = os.path.join(self.artifacts_path, filename)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
        print(f"Папка артефактов очищена ({self.artifacts_path})")

    # ============ Методы сохранения и загрузки ============

    def save_results(self):
        """Сохранение результатов обработки для train, test и scoring"""
        # Сохраняем train и test
        fmt = self.config["result"]["format"]
        save_path = self.config["result"]["save_path"]
        suffix = self.config["result"].get("suffix", "_preprocessed")

        if fmt == "parquet":
            # Для parquet save_path - это путь к директории
            ensure_dir_exists(save_path)
            train_output_path = f"{save_path}/train{suffix}.parquet"
            test_output_path = f"{save_path}/test{suffix}.parquet"

            if self.train_sdf is not None:
                self.save_data(self.train_sdf, fmt, train_output_path)
            if self.test_sdf is not None:
                self.save_data(self.test_sdf, fmt, test_output_path)
        elif fmt == "hive":
            # Для hive save_path - это имя таблицы
            train_table_name = f"{save_path}.train{suffix}"
            test_table_name = f"{save_path}.test{suffix}"

            if self.train_sdf is not None:
                self.save_data(self.train_sdf, fmt, train_table_name)
            if self.test_sdf is not None:
                self.save_data(self.test_sdf, fmt, test_table_name)
        else:
            raise ValueError(f"Неизвестный формат сохранения: {fmt}")

        # Сохраняем scoring
        if self.scoring_sdf is not None:
            fmt_scoring = self.config["result"].get("format_scoring", "parquet")
            save_path_scoring = self.config["result"].get("save_path_scoring", "")

            if fmt_scoring == "parquet":
                # Для parquet save_path_scoring - это путь к директории
                ensure_dir_exists(save_path_scoring)
                scoring_output_path = f"{save_path_scoring}/scoring{suffix}.parquet"
                self.save_data(self.scoring_sdf, fmt_scoring, scoring_output_path)
            elif fmt_scoring == "hive":
                # Для hive save_path_scoring - это имя таблицы
                scoring_table_name = f"{save_path_scoring}.scoring{suffix}"
                self.save_data(self.scoring_sdf, fmt_scoring, scoring_table_name)
            else:
                raise ValueError(f"Неизвестный формат сохранения для scoring: {fmt_scoring}")

            print(f"[INFO] Данные scoring сохранены (формат: {fmt_scoring})")

        print(f"Окончательный список признаков в файле: {self.path_features}")
        print(f"Окончательный список категориальных признаков в файле: {self.path_categorical}")

    def save_data(self, data, fmt, name):
        """Сохранение данных в указанном формате"""
        if fmt == "parquet":
            pdf = data.toPandas()
            ensure_dir_exists(os.path.dirname(name))
            pdf.to_parquet(name)
            print(f"[INFO] Данные сохранены в файл {name}")
        elif fmt == 'hive':
            self.sqlContext.sql(f'DROP TABLE IF EXISTS {name} PURGE')
            data.write.mode("overwrite").saveAsTable(name)
            print(f"[INFO] Данные сохранены в hive-таблицу {name}")
        else:
            raise ValueError(f"Неизвестный формат сохранения: {fmt}")

    def save_artifacts_info(self):
        """Сохранение информации об артефактах"""
        artifacts_json_path = os.path.join(self.artifacts_path, "artifacts_info.json")
        ensure_dir_exists(os.path.dirname(artifacts_json_path))
        with open(artifacts_json_path, "w", encoding='utf-8') as f:
            json.dump(self.artifacts_info, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Сохранена информация об артефактах")

    # Словарь методов для динамического вызова по имени шага
    preprocessing_methods = {
        "drop_forced_cols": drop_forced_cols,
        "convert_decimal": convert_decimal,
        "fill_missing": fill_missing,
        "drop_uninformative": drop_uninformative,
        "deduplicate": deduplicate,
        "categorical_encoding": categorical_encoding,
        "scaling": scaling,
        "drop_correlated": drop_correlated,
        "drop_adversarial": drop_adversarial,
        "noise_selection": select_noise,
        "forward_selection": select_forward,
        "permutation_selection": select_permutation
    }


if __name__ == '__main__':
    from campaignds.sc_connector import SparkHelper

    helper = SparkHelper(
        dynamicAllocation='True',
        name='preprocessing',
        instances=30,
        cores=3,
        spark_version='3.2.1',
        executor_memory='12g',
        driver_memory='16g',
        parcel='MPACK',
        mpack_name='sota'
    )

    config = {
        'spark.ui.showConsoleProgress': 'true',
        'hive.exec.dynamic.partition.mode': 'nonstrict',
        'spark.sql.sources.partitionOverwriteMode': 'dynamic',
        'spark.sql.hive.metastore.version': '2.3.9',
        'mapreduce.input.fileinputformat.input.dir.recursive': 'true',
    }
    sc, sqlContext = helper.create_sql_context(config)

    # Создаем экземпляр препроцессора с конфигурацией
    preprocessor = PreprocessingPipeline('my_config.yaml', sc, sqlContext)

    # Пример загрузки DataFrame для препроцессинга
    train_df = sqlContext.table("arnsdpsbx_t_team_cvm.train_result_202410_shadow_smz")
    valid_df = sqlContext.table("arnsdpsbx_t_team_cvm.eval_result_202411_shadow_smz")
    scoring_df = sqlContext.table("arnsdpsbx_t_team_cvm.scoring_data")

    # Запускаем препроцессинг с переданными DataFrame
    train_processed, valid_processed, scoring_processed = preprocessor.run_preprocessing(
        train_df, valid_df, scoring_df
    )
