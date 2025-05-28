import os
import numpy as np
import pandas as pd
import pyspark.sql.functions as F # type: ignore
import pyspark.sql.types as T # type: ignore
from pyspark.sql import DataFrame # type: ignore
import lightgbm as lgbm
from sklearn.metrics import roc_auc_score, mean_squared_error
from .base_feature_selector import BaseFeatureSelector


class NoiseFeatureSelector(BaseFeatureSelector):
    """
    Класс, реализующий алгоритм "noise selection":
    - Добавляет шумовой признак noise_feature
    - Разбивает train на фолды (fold_id) и train/val (is_train)
    - В каждом фолде обучает LightGBM, считает важности
    - Сравнивает важность признака с важностью noise_feature
    - Оставляет только те фичи, у которых в достаточно большом числе фолдов (≥ selection_rate) importance > noise_importance
    """

    def __init__(self, config, sc):
        self.config = config
        self.sc = sc
        self.noise_config = config["preprocessing"].get("noise_selection", {})
        self.artifacts_path = config["artifacts"]["path"]
        self.task_type = config["common"].get("task_type", "binary_classification")

        # Для удобства можно сразу вычитать некоторые параметры
        self.cv_folds = self.noise_config.get("cv_folds", 5)
        self.selection_rate = self.noise_config.get("selection_rate", 0.7)
        self.learning_rate = self.noise_config.get("learning_rate", 0.05)
        self.sample_size = self.noise_config.get("sample_size", None)
        self.num_boost_round = self.noise_config.get("num_boost_round", 100)
        self.seed = self.noise_config.get("seed", 42)
        self.importance_type = self.noise_config.get("importance_type", "gain")
        self.importances_file = f"{self.artifacts_path}/noise_importances.xlsx"

    def get_features_to_exclude(self, train_sdf, test_sdf=None, features=None, categorical_features=None):
        """
        Для noise selection используем «+фичи», поэтому список исключаемых фичей не нужен.
        """
        return []

    @staticmethod
    def _get_task_config(task_type):
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

    def get_features_to_include(self, train_sdf, test_sdf=None, features=None, categorical_features=None):
        """
        Основной метод:
        1) При необходимости сэмплируем train_sdf
        2) Добавляем fold_id (0..cv_folds-1) и is_train (0/1)
        3) Группируем по fold_id, внутри каждого фолда вызываем applyInPandas -> _train_on_fold(...)
        4) Собираем результаты, считаем для каждого признака, в скольких фолдах он был «важнее noise»
        5) Оставляем только те, у которых (count_above_noise / cv_folds) >= selection_rate
        """
        if not features:
            raise ValueError("[NoiseFeatureSelector] Пустой список фичей")

        # --- Шаг 1. При необходимости сэмплируем ---
        sdf = train_sdf
        total_cnt = sdf.count()
        if self.sample_size and self.sample_size < total_cnt:
            frac = float(self.sample_size) / total_cnt
            sdf = sdf.sample(withReplacement=False, fraction=frac, seed=42)
            print(f"[NoiseFeatureSelector] Сэмплируем train_sdf до ~{self.sample_size} строк")

        # --- Шаг 2. Генерация fold_id (0..cv_folds-1) и is_train (0 или 1) ---
        # Аналогично adversarial_feature_remover, но можно упрощённо:
        sdf = sdf.withColumn("fold_id", F.floor(F.rand(seed=42) * self.cv_folds))
        # Допустим, 70% строк в train, 30% в val внутри каждого фолда:
        sdf = sdf.withColumn("is_train", (F.rand(seed=43) > 0.3).cast("int"))

        task_config = NoiseFeatureSelector._get_task_config(self.task_type)

        # --- Шаг 3. applyInPandas ---
        train_func = self._get_train_on_fold_func(
            features=features,
            categorical_features=categorical_features,
            target_col=self.config["common"]["target_col"],
            importance_type=self.importance_type,
            num_boost_round=self.num_boost_round,
            learning_rate=self.learning_rate,
            task_config=task_config,
        )

        # Группируем по fold_id, запускаем обучение на каждой группе
        schema = "fold_id int, feature_name string, importance double, noise_important double, is_above_noise int"
        result_sdf = sdf.groupBy("fold_id").applyInPandas(train_func, schema=schema)
        result_pdf = result_sdf.toPandas()
        result_pdf.to_excel(self.importances_file, index_label='Num')
        print(f"[NoiseFeatureSelector] файл importances сохранен {result_pdf.shape}")

        # --- Шаг 4. Считаем долю фолдов, где фича оказалась важнее noise ---
        agg_pdf = result_pdf.groupby("feature_name")[["is_above_noise"]].sum().reset_index()
        agg_pdf.columns = ['feature_name', 'above_noise_cnt']

        agg_pdf["ratio"] = agg_pdf["above_noise_cnt"] / float(self.cv_folds)
        feats_to_keep = agg_pdf.loc[agg_pdf["ratio"] >= self.selection_rate, "feature_name"].tolist()

        print(f"[NoiseFeatureSelector] Всего фичей: {len(features)}")
        print(f"[NoiseFeatureSelector] Порог доли фолдов: {self.selection_rate}")
        print(f"[NoiseFeatureSelector] Оставляем фичей: {len(feats_to_keep)}")
        print(f"[NoiseFeatureSelector] Importance type: {self.importance_type}")
        return feats_to_keep

    @staticmethod
    def _get_train_on_fold_func(features, categorical_features, target_col, importance_type, num_boost_round, learning_rate, task_config):
        """
        Возвращаем статическую функцию для groupBy(...).applyInPandas(...).
        Она обучает LightGBM на (is_train==1), валидирует на (is_train==0),
        считает importance_gain, сравнивает с importance_gain шумовой фичи,
        и возвращает is_above_noise = 1/0.
        """
        def train_on_fold(pdf):
            fold_id = pdf["fold_id"].iloc[0]

            # Добавляем noise_feature
            np.random.seed(42+fold_id)
            pdf["noise_feature"] = np.random.random(size=len(pdf))
            # В итоге список признаков = исходные + noise_feature
            used_features = features + ["noise_feature"]

            # Разделяем train/val
            train_part = pdf[pdf["is_train"] == 1]
            val_part = pdf[pdf["is_train"] == 0]

            X_train = train_part[used_features]
            y_train = train_part[target_col]
            X_val = val_part[used_features]
            y_val = val_part[target_col]

            # Обучаем
            lgb_train = lgbm.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
            lgb_val = lgbm.Dataset(X_val, label=y_val, reference=lgb_train, categorical_feature=categorical_features)

            lgbm_params = {
                **task_config,
                'boosting_type': "gbdt",
                'verbosity': -1,
                'num_threads': -1,
                'seed': 42,
                'learning_rate': learning_rate,
                'max_depth': 16,
                'num_leaves': 240,
                'feature_fraction': 0.6,
                'bagging_fraction': 0.8,
                'bagging_freq': 1,
                'max_bin': 63,
                'min_child_weight': 10.0,
                'early_stopping_rounds': 10,
            }

            model = lgbm.train(
                lgbm_params,
                train_set=lgb_train,
                num_boost_round=num_boost_round,
                valid_sets=[lgb_val],
            )

            if importance_type == 'shap':
                shap_values_full = model.predict(X_val, pred_contrib=True)
                shap_values = shap_values_full[:, :-1]
                importances = np.mean(np.abs(shap_values), axis=0)
                feat_names = model.feature_name()
            else:
                importances = model.feature_importance(importance_type=importance_type)
                feat_names = model.feature_name()

            assert len(importances) == len(feat_names)

            df_imp = pd.DataFrame({"feature_name": feat_names, "importance": importances})

            # Смотрим, какая importance у noise_feature
            noise_imp = df_imp.loc[df_imp["feature_name"] == "noise_feature", "importance"]
            noise_imp_val = noise_imp.values[0] if len(noise_imp) else 0.0

            # Сравниваем c noise_importance
            df_imp["is_above_noise"] = (df_imp["importance"] > noise_imp_val).astype(int)
            # Исключаем сам noise_feature из финального вывода
            df_imp = df_imp[df_imp["feature_name"] != "noise_feature"]

            df_imp["noise_important"] = noise_imp_val
            df_imp["fold_id"] = fold_id
            return df_imp[["fold_id", "feature_name", "importance", "noise_important", "is_above_noise"]]

        return train_on_fold
