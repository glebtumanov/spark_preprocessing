import os
import numpy as np
import pandas as pd
import lightgbm as lgbm
from multiprocessing import Pool
import multiprocessing

import pyspark.sql.functions as F # type: ignore
import pyspark.sql.types as T # type: ignore
from sklearn.metrics import roc_auc_score, mean_squared_error

from .base_feature_selector import BaseFeatureSelector


def get_task_config(task_type):
    """Возвращает конфигурацию LightGBM и метрики в зависимости от типа задачи"""
    if task_type == "binary_classification":
        return {
            'objective': 'binary',
            'metric': 'auc'
        }
    elif task_type == "regression":
        return {
            'objective': 'regression',
            'metric': 'rmse'
        }
    else:
        raise ValueError(f"Неподдерживаемый тип задачи: {task_type}")


class CorrFeatureRemover(BaseFeatureSelector):
    def __init__(self, config, sc):
        """
        config: dict — это конфиг, который раньше хранился в self.config внутри PreprocessingPipeline
        sc: SparkContext (может понадобиться для broadcast)
        """
        self.config = config
        self.sc = sc
        self.feature_importances = {}
        self.feature_shap_values = {}

        self.n_jobs = self.config["common"].get("n_jobs", -1)
        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()

        self.imp_config = self.config["preprocessing"]["drop_correlated"]["importances"]
        self.task_type = self.config["common"].get("task_type", "binary_classification")

        self.use_shap = self.imp_config.get("use_shap", False)
        self.artifacts_path = self.config["artifacts"]["path"]
        self.importance_file = os.path.join(self.artifacts_path, f"feature_importances_by_{'shap' if self.use_shap else 'importances'}.csv")

    # --------------------
    # Методы для вычисления важностей (importances) и SHAP
    # --------------------

    @staticmethod
    def get_train_importances_func(features, target_col, lgb_params, cat_cols):
        """То же самое, что и раньше: train_on_partition_func для вычисления feature importances."""
        def train_on_partition(pdf):
            X = pdf[features]
            y = pdf[target_col]

            train_dataset = lgbm.Dataset(
                X,
                label=y,
                categorical_feature=cat_cols
            )

            model = lgbm.train(
                params=lgb_params,
                train_set=train_dataset,
            )
            importances = model.feature_importance(importance_type='gain')
            feature_names = model.feature_name()

            result_df = pd.DataFrame({
                'feature_name': feature_names,
                'importance_gain': importances,
                'fold_id': [pdf['fold_id'].iloc[0]] * len(feature_names)
            })
            return result_df
        return train_on_partition


    @staticmethod
    def get_train_shap_func(features, target_col, lgb_params, cat_cols):
        """Похоже на get_train_importances_func, только считаем SHAP."""
        def shap_on_partition(pdf):
            X = pdf[features]
            y = pdf[target_col]

            train_dataset = lgbm.Dataset(
                X,
                label=y,
                categorical_feature=cat_cols
            )

            model = lgbm.train(
                params=lgb_params,
                train_set=train_dataset,
            )

            shap_values_full = model.predict(X, pred_contrib=True)
            shap_values = shap_values_full[:, :-1]  # убираем base value
            shap_means = np.mean(np.abs(shap_values), axis=0)

            feature_names = model.feature_name()
            fold_id = pdf['fold_id'].iloc[0] if len(pdf) > 0 else -1

            result_df = pd.DataFrame({
                'feature_name': feature_names,
                'shap_value': shap_means,
                'fold_id': [fold_id] * len(feature_names)
            })
            return result_df
        return shap_on_partition


    def get_feature_importances(self, train_sdf, features, cat_cols):
        """
        Вычисляет LightGBM-importances (gain) через разбивку на n_partitions.
        Сохраняет результат в self.feature_importances.
        Возвращает dict: {feature_name -> importance}.
        """
        n_partitions = self.imp_config["n_partitions"]
        sample_size = self.imp_config["sample_size"]
        learning_rate = self.imp_config["learning_rate"]

        # Сэмплирование, если нужно
        if sample_size and int(sample_size) > 0:
            total_cnt = train_sdf.count()
            if sample_size < total_cnt:
                fraction = sample_size / float(total_cnt)
                train_sdf = train_sdf.sample(False, fraction, seed=42)
        # fold_id
        train_sdf = train_sdf.withColumn("fold_id", F.floor(F.rand(seed=42) * n_partitions))

        # Получаем конфигурацию для типа задачи
        task_config = get_task_config(self.task_type)

        # Параметры LightGBM
        lgb_params = {
            'num_boost_round': 200,
            'objective': task_config['objective'],
            'metric': task_config['metric'],
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'num_threads': -1,
            'seed': 1,
            'learning_rate': learning_rate,
            'max_depth': 16,
            'num_leaves': 244,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'max_bin': 63
        }

        train_on_partition_func = self.get_train_importances_func(
            features,
            self.config["common"]["target_col"],
            lgb_params,
            cat_cols,
        )

        schema = "feature_name string, importance_gain double, fold_id int"
        importances_sdf = (
            train_sdf.groupBy("fold_id")
            .applyInPandas(train_on_partition_func, schema=schema)
        )

        # Усредняем
        avg_importances_sdf = (
            importances_sdf.groupBy("feature_name")
            .agg(F.avg("importance_gain").alias("importance_gain"))
        )

        # В словарь
        importances_dict = {}
        for row in avg_importances_sdf.collect():
            importances_dict[row["feature_name"]] = row["importance_gain"]

        self.feature_importances = importances_dict
        return self.feature_importances

    def get_feature_shap_values(self, train_sdf, features, cat_cols):
        """
        Аналогично get_feature_importances, но считаем SHAP и усредняем.
        Сохраняем в self.feature_shap_values.
        """
        n_partitions = self.imp_config["n_partitions"]
        sample_size = self.imp_config["sample_size"]
        learning_rate = self.imp_config["learning_rate"]

        if sample_size and sample_size > 0:
            total_cnt = train_sdf.count()
            if sample_size < total_cnt:
                fraction = sample_size / float(total_cnt)
                train_sdf = train_sdf.sample(False, fraction, seed=42)

        train_sdf = train_sdf.withColumn("fold_id", F.floor(F.rand(seed=42) * n_partitions))

        # Получаем конфигурацию для типа задачи
        task_config = get_task_config(self.task_type)

        lgb_params = {
            'num_boost_round': 200,
            'objective': task_config['objective'],
            'metric': task_config['metric'],
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'num_threads': -1,
            'seed': 1,
            'learning_rate': learning_rate,
            'max_depth': 16,
            'num_leaves': 244,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'max_bin': 63
        }

        shap_on_partition_func = self.get_train_shap_func(
            features,
            self.config["common"]["target_col"],
            lgb_params,
            cat_cols,
        )

        schema = "feature_name string, shap_value double, fold_id int"
        shap_sdf = (
            train_sdf.groupBy("fold_id")
            .applyInPandas(shap_on_partition_func, schema=schema)
        )

        avg_shap_sdf = (
            shap_sdf.groupBy("feature_name")
            .agg(F.avg("shap_value").alias("shap_value"))
        )

        shap_values_dict = {}
        for row in avg_shap_sdf.collect():
            shap_values_dict[row["feature_name"]] = row["shap_value"]

        self.feature_shap_values = shap_values_dict
        return self.feature_shap_values

    def get_feature_scores(self, train_sdf, features, cat_cols):
        """
        Если use_shap=True, считаем shap, иначе importance_gain.
        """
        if self.use_shap:
            return self.get_feature_shap_values(train_sdf, features, cat_cols)
        else:
            return self.get_feature_importances(train_sdf, features, cat_cols)

    # --------------------
    # Корреляция
    # --------------------

    @staticmethod
    def _compute_corr(args):
        data, f1, f2 = args
        c = np.corrcoef(data[f1].values, data[f2].values)[0, 1]
        return f1, f2, c

    def get_correlated_features(self, train_sdf, features, importances_dict):
        """
        Поиск фичей с корреляцией >= threshold.
        Возвращает список фичей, которые стоит исключить (исходя из важности).
        """
        threshold = self.config["preprocessing"]["drop_correlated"]["threshold"]
        sample_size = self.config["preprocessing"]["drop_correlated"]["sample_size"]

        # Собираем сэмпл
        if sample_size and int(sample_size) > 0:
            pdf_sample = self._sample_for_fit(train_sdf, features, sample_size)
        else:
            pdf_sample = train_sdf.toPandas()

        if pdf_sample.empty:
            return []

        # Формируем пары (f1, f2)
        pairs = []
        numeric_feats = pdf_sample.columns.tolist()
        for i, f1 in enumerate(numeric_feats):
            for f2 in numeric_feats[i+1:]:
                pairs.append((pdf_sample, f1, f2))

        # Параллельный расчёт
        with Pool(self.n_jobs) as p:
            results = p.map(self._compute_corr, pairs)

        # Отбираем пары с высокой корреляцией
        high_corr_pairs = [(f1, f2) for f1, f2, c in results if abs(c) >= threshold]

        feats_to_remove = []
        for f1, f2 in high_corr_pairs:
            imp1 = importances_dict.get(f1, 0)
            imp2 = importances_dict.get(f2, 0)
            if imp1 < imp2:
                feats_to_remove.append(f1)
            else:
                feats_to_remove.append(f2)

        return list(set(feats_to_remove))

    # --------------------
    # Реализация абстрактных методов BaseFeatureSelector
    # --------------------

    def get_features_to_exclude(self, train_sdf, test_sdf=None, features=None, categorical_features=None):
        """
        1) Определяем список фичей (train_sdf.columns без index_col/target/skip и т.д.)
        2) Считаем важности или shap
        3) Находим коррелирующиеся фичи
        4) Возвращаем их списком
        """
        index_col = self.config["common"]["index_col"]
        target_col = self.config["common"]["target_col"]
        skip_cols = set(self.config["common"].get("skip_cols", []))

        # Собираем cat_cols, num_cols (логика может быть чуть другой, как в твоём коде)
        cat_cols, num_cols = self._split_cols_by_type(train_sdf, index_col, target_col, skip_cols)
        features = cat_cols + num_cols
        if not features:
            return []

        # Считаем importances_dict (или shap)
        importances_dict = self.get_feature_scores(train_sdf, features, cat_cols)
        if not importances_dict:
            # Если не получилось посчитать — возвращаем пустой список
            return []

        imp_df = pd.DataFrame([(k, v) for k, v in importances_dict.items()], columns=['Feature', 'Importance'])
        imp_df.sort_values('Importance', ascending=False).to_csv(self.importance_file, index=False)
        print(f"[CorrFeatureRemover] Файл с feature importances сохранен, метод: {'shap' if self.use_shap else 'importances'}")

        # Ищем высококоррелированные фичи
        feats_to_remove = self.get_correlated_features(train_sdf, list(importances_dict.keys()), importances_dict)
        return feats_to_remove

    def get_features_to_include(self, train_sdf):
        """
        Для этого класса логики «принудительно включать» нет, возвращаем пустой список.
        """
        return []

    # --------------------
    # Дополнительные методы
    # --------------------

    @staticmethod
    def _sample_for_fit(sdf, cols, sample_size=10000):
        cnt = sdf.count()
        if cnt == 0:
            return pd.DataFrame(columns=cols)

        fraction = 1.0
        if sample_size and sample_size < cnt:
            fraction = sample_size / float(cnt)

        sdf_sample = sdf.sample(False, fraction, seed=42)
        if sample_size:
            sdf_sample = sdf_sample.limit(sample_size)

        pdf = sdf_sample.select(*cols).toPandas()
        return pdf


    @staticmethod
    def _split_cols_by_type(sdf, index_col, target_col, skip_cols):
        cat_cols = []
        num_cols = []
        for field in sdf.schema.fields:
            col_name = field.name
            if col_name in [index_col, target_col, "split_group"]:
                continue
            if col_name in skip_cols:
                continue
            spark_type = field.dataType
            # Упрощённая логика — если StringType -> cat, иначе -> num
            # (У тебя она чуть сложнее, см. исходный код.)
            if str(spark_type) == "StringType":
                cat_cols.append(col_name)
            else:
                num_cols.append(col_name)
        return cat_cols, num_cols
