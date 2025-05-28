import numpy as np
import pandas as pd
import pyspark.sql.functions as F # type: ignore
import pyspark.sql.types as T # type: ignore
from pyspark.sql import DataFrame # type: ignore
import lightgbm as lgbm
from sklearn.metrics import roc_auc_score, mean_squared_error
from .base_feature_selector import BaseFeatureSelector


class PermutationSelector(BaseFeatureSelector):
    """
    Класс, реализующий алгоритм "permutation importance selection":
    - Разбивает train на фолды (fold_id) и train/val (is_train)
    - В каждом фолде обучает LightGBM на всех фичах
    - Для каждой фичи вычисляет permutation importance:
      * Получает базовую метрику на валидации
      * Перемешивает значения фичи и пересчитывает метрику
      * Важность = разность метрик (с учётом направления оптимизации)
    - Оставляет только те фичи, у которых в достаточно большом числе фолдов (≥ selection_rate)
      permutation importance > min_importance_threshold
    """

    def __init__(self, config, sc):
        self.config = config
        self.sc = sc
        self.perm_config = config["preprocessing"].get("permutation_selection", {})
        self.artifacts_path = config["artifacts"]["path"]
        self.task_type = config["common"].get("task_type", "binary_classification")

        # Параметры
        self.cv_folds = self.perm_config.get("cv_folds", 5)
        self.selection_rate = self.perm_config.get("selection_rate", 0.7)
        self.min_importance_threshold = self.perm_config.get("min_importance_threshold", 0.001)
        self.learning_rate = self.perm_config.get("learning_rate", 0.05)
        self.sample_size = self.perm_config.get("sample_size", None)
        self.num_boost_round = self.perm_config.get("num_boost_round", 100)
        self.seed = self.perm_config.get("seed", 42)
        self.valid_frac = self.perm_config.get("valid_frac", 0.3)
        self.permutation_repeats = self.perm_config.get("permutation_repeats", 1)
        self.importances_file = f"{self.artifacts_path}/permutation_importances.xlsx"

    def get_features_to_exclude(self, train_sdf, test_sdf=None, features=None, categorical_features=None):
        """
        Для permutation selection используем «+фичи», поэтому список исключаемых фичей не нужен.
        """
        return []

    @staticmethod
    def _get_task_config(task_type):
        """Возвращает конфигурацию LightGBM и метрики в зависимости от типа задачи"""
        if task_type == "binary_classification":
            return {
                'objective': 'binary',
                'metric': 'auc',
                'eval_metric': roc_auc_score,
                'higher_better': True  # Для AUC больше = лучше
            }
        elif task_type == "regression":
            return {
                'objective': 'regression',
                'metric': 'rmse',
                'eval_metric': lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False),
                'higher_better': False  # Для RMSE меньше = лучше
            }
        else:
            raise ValueError(f"Неподдерживаемый тип задачи: {task_type}")

    def get_features_to_include(self, train_sdf, test_sdf=None, features=None, categorical_features=None):
        """
        Основной метод:
        1) При необходимости сэмплируем train_sdf
        2) Добавляем fold_id (0..cv_folds-1) и is_train (0/1)
        3) Группируем по fold_id, внутри каждого фолда вызываем applyInPandas -> _train_on_fold(...)
        4) Собираем результаты permutation importance для каждого признака
        5) Оставляем только те, у которых (count_above_threshold / cv_folds) >= selection_rate
        """
        if not features:
            raise ValueError("[PermutationSelector] Пустой список фичей")

        # --- Шаг 1. При необходимости сэмплируем ---
        sdf = train_sdf
        total_cnt = sdf.count()
        if self.sample_size and self.sample_size < total_cnt:
            frac = float(self.sample_size) / total_cnt
            sdf = sdf.sample(withReplacement=False, fraction=frac, seed=42)
            print(f"[PermutationSelector] Сэмплируем train_sdf до ~{self.sample_size} строк")

        # --- Шаг 2. Генерация fold_id (0..cv_folds-1) и is_train (0 или 1) ---
        sdf = sdf.withColumn("fold_id", F.floor(F.rand(seed=42) * self.cv_folds))
        sdf = sdf.withColumn("is_train", (F.rand(seed=43) > self.valid_frac).cast("int"))

        task_config = PermutationSelector._get_task_config(self.task_type)

        # --- Шаг 3. applyInPandas ---
        train_func = self._get_train_on_fold_func(
            features=features,
            categorical_features=categorical_features,
            target_col=self.config["common"]["target_col"],
            num_boost_round=self.num_boost_round,
            learning_rate=self.learning_rate,
            task_config=task_config,
            permutation_repeats=self.permutation_repeats,
            min_importance_threshold=self.min_importance_threshold,
        )

        # Группируем по fold_id, запускаем обучение на каждой группе
        schema = "fold_id int, feature_name string, permutation_importance double, is_above_threshold int"
        result_sdf = sdf.groupBy("fold_id").applyInPandas(train_func, schema=schema)
        result_pdf = result_sdf.toPandas()
        result_pdf.to_excel(self.importances_file, index_label='Num')
        print(f"[PermutationSelector] файл importances сохранен {result_pdf.shape}")

        # --- Шаг 4. Считаем долю фолдов, где фича оказалась важной ---
        agg_pdf = result_pdf.groupby("feature_name")[["is_above_threshold"]].sum().reset_index()
        agg_pdf.columns = ['feature_name', 'above_threshold_cnt']

        agg_pdf["ratio"] = agg_pdf["above_threshold_cnt"] / float(self.cv_folds)
        feats_to_keep = agg_pdf.loc[agg_pdf["ratio"] >= self.selection_rate, "feature_name"].tolist()

        print(f"[PermutationSelector] Всего фичей: {len(features)}")
        print(f"[PermutationSelector] Порог доли фолдов: {self.selection_rate}")
        print(f"[PermutationSelector] Минимальный порог важности: {self.min_importance_threshold}")
        print(f"[PermutationSelector] Оставляем фичей: {len(feats_to_keep)}")
        return feats_to_keep

    @staticmethod
    def _get_train_on_fold_func(features, categorical_features, target_col, num_boost_round, learning_rate, task_config, permutation_repeats, min_importance_threshold):
        """
        Возвращаем статическую функцию для groupBy(...).applyInPandas(...).
        Она обучает LightGBM на (is_train==1), валидирует на (is_train==0),
        вычисляет permutation importance для каждой фичи и сравнивает с порогом.
        """
        def train_on_fold(pdf):
            fold_id = pdf["fold_id"].iloc[0]

            # Разделяем train/val
            train_part = pdf[pdf["is_train"] == 1]
            val_part = pdf[pdf["is_train"] == 0]

            if train_part.empty or val_part.empty:
                raise ValueError(f"[train_on_fold] Недостаточно данных для фолда {fold_id}")

            X_train = train_part[features]
            y_train = train_part[target_col]
            X_val = val_part[features]
            y_val = val_part[target_col]

            # Обучаем базовую модель
            lgb_train = lgbm.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
            lgb_val = lgbm.Dataset(X_val, label=y_val, reference=lgb_train, categorical_feature=categorical_features)

            lgbm_params = {
                'objective': task_config['objective'],
                'metric': task_config['metric'],
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

            # Получаем базовую метрику
            baseline_pred = model.predict(X_val)
            baseline_score = task_config['eval_metric'](y_val, baseline_pred)

            # Вычисляем permutation importance для каждой фичи
            results = []

            for feature in features:
                importance_scores = []

                for _ in range(permutation_repeats):
                    # Создаем копию валидационных данных
                    X_val_perm = X_val.copy()

                    # Перемешиваем значения текущей фичи
                    np.random.seed(42 + fold_id + hash(feature) % 1000)
                    X_val_perm[feature] = np.random.permutation(X_val_perm[feature].values)

                    # Получаем предсказания с перемешанной фичей
                    perm_pred = model.predict(X_val_perm)
                    perm_score = task_config['eval_metric'](y_val, perm_pred)

                    # Вычисляем важность с учётом направления оптимизации
                    if task_config['higher_better']:
                        # Для метрик типа AUC (больше = лучше)
                        importance = baseline_score - perm_score
                    else:
                        # Для метрик типа RMSE (меньше = лучше)
                        importance = perm_score - baseline_score

                    importance_scores.append(importance)

                # Усредняем importance по повторениям
                avg_importance = np.mean(importance_scores)
                is_above_threshold = int(avg_importance > min_importance_threshold)

                results.append({
                    "fold_id": fold_id,
                    "feature_name": feature,
                    "permutation_importance": avg_importance,
                    "is_above_threshold": is_above_threshold
                })

            return pd.DataFrame(results)

        return train_on_fold