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
from abc import ABC, abstractmethod

# pyspark
import pyspark # type: ignore
from pyspark.sql import SparkSession, DataFrame # type: ignore
from pyspark.sql import functions as F # type: ignore
from pyspark.sql import types as T # type: ignore
from pyspark.sql.functions import pandas_udf, col # type: ignore
from pyspark.ml.feature import StringIndexer, StringIndexerModel # type: ignore
from pyspark.ml.feature import VectorAssembler, StandardScaler, StandardScalerModel, MinMaxScaler, MinMaxScalerModel # type: ignore
from pyspark.ml import Pipeline # type: ignore

# sklearn
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
    KBinsDiscretizer
)
from sklearn.metrics import roc_auc_score

# category_encoders
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.cat_boost import CatBoostEncoder

# локальные модули
from selectors.corr_feature_remover import CorrFeatureRemover
from selectors.adversarial_feature_remover import AdversarialFeatureRemover
from selectors.noise_feature_selector import NoiseFeatureSelector
from selectors.forward_selection import ForwardFeatureSelector

sklearn.set_config(transform_output='pandas')

# Включаем поддержку Arrow для ускорения передачи данных между Spark и pandas
spark_conf = pyspark.SparkConf()
spark_conf.set('spark.sql.execution.arrow.enabled', 'true')

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


class FeatureSelector(ABC):
    """
    Базовый интерфейс для классов, которые должны определять,
    какие фичи стоит исключить или, наоборот, оставить.
    """

    @abstractmethod
    def get_features_to_exclude(self, train_sdf, test_sdf=None, features=None, categorical_features=None):
        """
        Возвращает список фичей, которые необходимо исключить.
        """
        pass

    @abstractmethod
    def get_features_to_include(self, train_sdf, test_sdf=None, features=None, categorical_features=None):
        """
        Возвращает список фичей, которые необходимо включить.
        """
        pass


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

        # Основные настройки
        self.index_col = self.config["common"]["index_col"]
        self.target_col = self.config["common"].get("target_col")
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
                method = self.preprocessing_methods[step]
                method(self)
            else:
                print(f"[WARN] Неизвестный шаг пайплайна: {step}")

        # Сохранение информации и результатов
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
            print(f"[Artefact] Список удалённых колонок сохранён: {path}")

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
            print(f"[Artefact] Список неинформативных колонок сохранён: {path}")

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
        if self.scoring_sdf is not None:
            initial_count = self.scoring_sdf.count()
            self.scoring_sdf = self.scoring_sdf.dropDuplicates(subset=subset)
            final_count = self.scoring_sdf.count()
            removed = initial_count - final_count

            print(f"[deduplicate] Удалено {removed} строк из scoring")

    def categorical_encoding(self):
        """Кодирование категориальных признаков"""
        self._check_data_loaded()

        method = self.config["preprocessing"]["categorical_encoding"]["method"]
        cat_cols = list(self.cat_cols)

        if not cat_cols:
            return

        encoder_path = os.path.join(self.artifacts_path, "encoder.pkl")

        sample_size = self.config["preprocessing"]["categorical_encoding"].get("sample_size")

        if sample_size:
            train_sample_pdf = self._sample_for_fit(self.train_sdf, cat_cols, sample_size=sample_size)
        else:
            # Используем Arrow для toPandas
            train_sample_pdf = self.train_sdf.toPandas()

        if method == "label_encoder":
            # Используем Spark ML StringIndexer
            stages = []
            for c in cat_cols:
                stages.append(StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep"))
            pipeline = Pipeline(stages=stages)
            model = pipeline.fit(self.train_sdf)
            model.write().overwrite().save(encoder_path)
            print(f"[Artefact] Spark StringIndexer сохранён: {encoder_path}")
            # Применяем к train/test/scoring
            for attr in ["train_sdf", "test_sdf", "scoring_sdf"]:
                sdf = getattr(self, attr)
                if sdf is not None:
                    sdf = model.transform(sdf)
                    # Заменяем оригинальные колонки на индексированные
                    for c in cat_cols:
                        sdf = sdf.drop(c).withColumnRenamed(f"{c}_index", c)
                    setattr(self, attr, sdf)
        elif method == "target_encoder":
            self.encoder = TargetEncoder(min_samples_leaf=100, smoothing=10.0)
            self.encoder.fit(train_sample_pdf[cat_cols], train_sample_pdf[self.target_col])
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.encoder, f)
            print(f"[Artefact] TargetEncoder сохранён: {encoder_path}")
            # Применяем через pandas_udf
            func = get_target_encoder_udf(cat_cols, encoder=self.encoder)
            for attr in ["train_sdf", "test_sdf", "scoring_sdf"]:
                sdf = getattr(self, attr)
                if sdf is not None:
                    fields = [T.StructField(f.name, T.FloatType(), True) if f.name in cat_cols else f for f in sdf.schema.fields]
                    schema = T.StructType(fields)
                    sdf = sdf.mapInPandas(func, schema=schema)
                    setattr(self, attr, sdf)
            # Переводим cat_cols в num_cols
            for c in cat_cols:
                if c in self.cat_cols:
                    self.cat_cols.remove(c)
                if c not in self.num_cols:
                    self.num_cols.append(c)
        elif method == "catboost_encoder":
            self.encoder = CatBoostEncoder(cols=cat_cols)
            self.encoder.fit(train_sample_pdf[cat_cols], train_sample_pdf[self.target_col])
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.encoder, f)
            print(f"[Artefact] CatBoostEncoder сохранён: {encoder_path}")
            func = get_target_encoder_udf(cat_cols, encoder=self.encoder)
            for attr in ["train_sdf", "test_sdf", "scoring_sdf"]:
                sdf = getattr(self, attr)
                if sdf is not None:
                    fields = [T.StructField(f.name, T.FloatType(), True) if f.name in cat_cols else f for f in sdf.schema.fields]
                    schema = T.StructType(fields)
                    sdf = sdf.mapInPandas(func, schema=schema)
                    setattr(self, attr, sdf)
            for c in cat_cols:
                if c in self.cat_cols:
                    self.cat_cols.remove(c)
                if c not in self.num_cols:
                    self.num_cols.append(c)
        else:
            raise ValueError(f"Неизвестный метод кодирования: {method}")

    def scaling(self):
        """Масштабирование числовых признаков"""
        self._check_data_loaded()

        num_cols = list(self.num_cols)
        if not num_cols:
            return

        scaler_path = os.path.join(self.artifacts_path, "scaler.pkl")

        sample_size = self.config["preprocessing"]["scaling"].get("sample_size")

        if sample_size:
            train_sample_pdf = self._sample_for_fit(self.train_sdf, num_cols, sample_size=sample_size)
        else:
            train_sample_pdf = self.train_sdf.toPandas()

        method = self.config["preprocessing"]["scaling"]["method"]

        if method == "standard":
            # Spark ML StandardScaler
            assembler = VectorAssembler(inputCols=num_cols, outputCol="features_vec")
            scaler = StandardScaler(inputCol="features_vec", outputCol="scaled_features", withMean=True, withStd=True)
            pipeline = Pipeline(stages=[assembler, scaler])
            model = pipeline.fit(self.train_sdf)
            model.write().overwrite().save(scaler_path)
            print(f"[Artefact] Spark StandardScaler сохранён: {scaler_path}")
            for attr in ["train_sdf", "test_sdf", "scoring_sdf"]:
                sdf = getattr(self, attr)
                if sdf is not None:
                    sdf = model.transform(sdf)
                    # Разворачиваем scaled_features обратно в колонки
                    for i, c in enumerate(num_cols):
                        sdf = sdf.withColumn(c, col("scaled_features")[i])
                    sdf = sdf.drop("features_vec", "scaled_features")
                    setattr(self, attr, sdf)
        elif method == "minmax":
            assembler = VectorAssembler(inputCols=num_cols, outputCol="features_vec")
            scaler = MinMaxScaler(inputCol="features_vec", outputCol="scaled_features")
            pipeline = Pipeline(stages=[assembler, scaler])
            model = pipeline.fit(self.train_sdf)
            model.write().overwrite().save(scaler_path)
            print(f"[Artefact] Spark MinMaxScaler сохранён: {scaler_path}")
            for attr in ["train_sdf", "test_sdf", "scoring_sdf"]:
                sdf = getattr(self, attr)
                if sdf is not None:
                    sdf = model.transform(sdf)
                    for i, c in enumerate(num_cols):
                        sdf = sdf.withColumn(c, col("scaled_features")[i])
                    sdf = sdf.drop("features_vec", "scaled_features")
                    setattr(self, attr, sdf)
        elif method == "quantile":
            self.scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
            self.scaler.fit(train_sample_pdf[num_cols])
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"[Artefact] QuantileTransformer сохранён: {scaler_path}")
            transform_func = get_transform_udf(num_cols, scaler=self.scaler)
            for attr in ["train_sdf", "test_sdf", "scoring_sdf"]:
                sdf = getattr(self, attr)
                if sdf is not None:
                    fields = [T.StructField(f.name, T.FloatType(), True) if f.name in num_cols else f for f in sdf.schema.fields]
                    schema = T.StructType(fields)
                    sdf = sdf.mapInPandas(transform_func, schema=schema)
                    setattr(self, attr, sdf)
        elif method == "binning":
            from sklearn.preprocessing import KBinsDiscretizer
            bins = self.config["preprocessing"]["scaling"].get("bins", 10)
            self.scaler = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
            self.scaler.fit(train_sample_pdf[num_cols])
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"[Artefact] KBinsDiscretizer сохранён: {scaler_path}")
            transform_func = get_transform_udf(num_cols, scaler=self.scaler)
            for attr in ["train_sdf", "test_sdf", "scoring_sdf"]:
                sdf = getattr(self, attr)
                if sdf is not None:
                    fields = [T.StructField(f.name, T.FloatType(), True) if f.name in num_cols else f for f in sdf.schema.fields]
                    schema = T.StructType(fields)
                    sdf = sdf.mapInPandas(transform_func, schema=schema)
                    setattr(self, attr, sdf)
        else:
            raise ValueError(f"Неизвестный метод масштабирования: {method}")

    def drop_correlated(self):
        """Удаление сильно коррелирующих признаков"""
        self._check_data_loaded()
        selector = CorrFeatureRemover(self.config, self.sc)

        # Получаем список признаков для исключения
        features_to_exclude = selector.get_features_to_exclude(train_sdf=self.train_sdf)

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
            print(f"[Artefact] Список коррелирующих колонок сохранён: {path}")

            # Обновляем списки колонок
            self._remove_cols_from_lists(features_to_exclude)

    def drop_adversarial(self):
        """Удаление adversarial признаков - ВАЖНО: scoring_sdf не участвует в анализе"""
        self._check_data_loaded()
        selector = AdversarialFeatureRemover(self.config, self.sc)

        # Получаем список признаков для исключения (scoring не участвует)
        features_to_exclude = selector.get_features_to_exclude(
            train_sdf=self.train_sdf,
            test_sdf=self.test_sdf,
            features=self.final_features_list,
            categorical_features=self.cat_cols
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
            print(f"[Artefact] Список adversarial колонок сохранён: {path}")

            # Обновляем списки колонок
            self._remove_cols_from_lists(features_to_exclude)

    def select_noise(self):
        """Отбор признаков на основе шума"""
        self._check_data_loaded()
        selector = NoiseFeatureSelector(self.config, self.sc)

        # Получаем список признаков для включения
        features_to_include = selector.get_features_to_include(
            train_sdf=self.train_sdf,
            test_sdf=None,
            features=self.final_features_list,
            categorical_features=self.cat_cols
        )

        # Получаем список признаков для исключения
        features_to_exclude = [col for col in self.final_features_list if col not in features_to_include]

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
            print(f"[Artefact] Список noise-исключённых колонок сохранён: {path}")

            # Обновляем списки колонок
            self._remove_cols_from_lists(features_to_exclude)

    def select_forward(self):
        """Отбор признаков с помощью прямого отбора"""
        self._check_data_loaded()
        selector = ForwardFeatureSelector(self.config, self.sc)

        # Получаем список признаков для включения
        features_to_include = selector.get_features_to_include(
            train_sdf=self.train_sdf,
            test_sdf=None,
            features=self.final_features_list,
            categorical_features=self.cat_cols
        )

        # Получаем список признаков для исключения
        features_to_exclude = [col for col in self.final_features_list if col not in features_to_include]

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
            print(f"[Artefact] Список fs-исключённых колонок сохранён: {path}")

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

    def _remove_cols_from_lists(self, cols_to_remove):
        """Удаляет колонки из списков и сохраняет обновленные списки"""
        for c in cols_to_remove:
            if c in self.cat_cols:
                self.cat_cols.remove(c)
            if c in self.num_cols:
                self.num_cols.remove(c)
            if c in self.final_features_list:
                self.final_features_list.remove(c)

        # Сохраняем обновленные списки
        self.path_features = os.path.join(self.artifacts_path, "final_feature_list.txt")
        self.path_categorical = os.path.join(self.artifacts_path, "final_categorical_list.txt")

        ensure_dir_exists(os.path.dirname(self.path_features))
        ensure_dir_exists(os.path.dirname(self.path_categorical))

        with open(self.path_features, 'w') as f:
            f.write('\n'.join(self.final_features_list))
        print(f"[Artefact] Финальный список признаков сохранён: {self.path_features}")
        with open(self.path_categorical, 'w') as f:
            f.write('\n'.join(self.cat_cols))
        print(f"[Artefact] Финальный список категориальных признаков сохранён: {self.path_categorical}")

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
        fmt = self.config["result"]["format"]
        train_path = self.config["result"].get("train_save_path")
        test_path = self.config["result"].get("test_save_path")
        scoring_path = self.config["result"].get("scoring_save_path")

        if fmt == "parquet":
            if self.train_sdf is not None and train_path:
                ensure_dir_exists(os.path.dirname(train_path))
                self.save_data(self.train_sdf, fmt, train_path)
            if self.test_sdf is not None and test_path:
                ensure_dir_exists(os.path.dirname(test_path))
                self.save_data(self.test_sdf, fmt, test_path)
        elif fmt == "hive":
            if self.train_sdf is not None and train_path:
                self.save_data(self.train_sdf, fmt, train_path)
            if self.test_sdf is not None and test_path:
                self.save_data(self.test_sdf, fmt, test_path)
        else:
            raise ValueError(f"Неизвестный формат сохранения: {fmt}")

        if self.scoring_sdf is not None and scoring_path:
            fmt_scoring = self.config["result"].get("format_scoring", fmt)
            if fmt_scoring == "parquet":
                ensure_dir_exists(os.path.dirname(scoring_path))
                self.save_data(self.scoring_sdf, fmt_scoring, scoring_path)
            elif fmt_scoring == "hive":
                self.save_data(self.scoring_sdf, fmt_scoring, scoring_path)
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
            data.write.mode("overwrite").saveAsTable(name)
            print(f"[INFO] Данные сохранены в hive-таблицу {name}")
        else:
            raise ValueError(f"Неизвестный формат сохранения: {fmt}")

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
        "forward_selection": select_forward
    }


if __name__ == '__main__':
    from campaignds.sc_connector import SparkHelper # type: ignore

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
