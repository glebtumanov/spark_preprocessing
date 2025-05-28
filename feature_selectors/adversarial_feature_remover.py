import json
import time
import multiprocessing
import datetime as dt
import pandas as pd
import numpy as np
import lightgbm as lgbm

import pyspark.sql.functions as F # type: ignore
from pyspark.sql import DataFrame # type: ignore
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType # type: ignore

from .base_feature_selector import BaseFeatureSelector


class AdversarialFeatureRemover(BaseFeatureSelector):
    """
    Класс, реализующий adversarial feature removal (adversarial selection)
    на PySpark DataFrame и LightGBM в pandas_udf.
    """

    def __init__(self, config: dict, sc):
        """
        config: dict — общий конфиг пайплайна
        sc: SparkContext (для возможного использования broadcast, если нужно)
        """
        self.config = config
        self.sc = sc

        # Число потоков
        self.n_jobs = self.config["common"].get("n_jobs", -1)
        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()

        # Артефакты
        self.artifacts_path = self.config["artifacts"]["path"]

        # Параметры алгоритма из конфига
        self.drop_adv_config = self.config["preprocessing"].get("drop_adversarial", {})
        self.sample_size = self.drop_adv_config.get("sample_size", None)
        self.target_metric_value = self.drop_adv_config.get("target_metric_value", 0.75)
        self.cv_folds = self.drop_adv_config.get("cv_folds", 5)
        self.learning_rate = self.drop_adv_config.get("learning_rate", 0.05)

        self.max_features_to_remove = self.drop_adv_config.get("max_features_to_remove", 50)
        self.max_time = self.drop_adv_config.get("max_time", 3600)  # в секундах
        self.num_boost_round = self.drop_adv_config.get("num_boost_round", 100)
        self.importance_type = self.drop_adv_config.get("importance_type", "gain")
        self.removed_feature_file = f"{self.artifacts_path}/adversarial_removed_features.json"

    def get_features_to_exclude(
        self,
        train_sdf: DataFrame,
        test_sdf: DataFrame = None,
        features: list = None,
        categorical_features: list = None
    ) -> list:
        """
        Вычисляет список фичей, которые нужно исключить по методу adversarial selection.

        Параметры:
        ----------
        train_sdf : pyspark.sql.DataFrame
            Тренировочный датасет.
        test_sdf : pyspark.sql.DataFrame
            Тестовый (out-of-fold или OOT) датасет.
        features : list
            Список фичей, которые будем проверять на adversarial эффект.
        categorical_features : list
            Подсписок фичей из `features`, которые являются категориальными.

        Возвращает:
        -----------
        list:
            Список фичей, которые нужно исключить.
        """

        # 1) Проверки на входные данные
        if test_sdf is None:
            raise ValueError("Не передан test_sdf. Для adversarial selection нужен train и test.")
        if not features:
            raise ValueError("Список features пуст или не передан.")
        if categorical_features is None:
            categorical_features = []

        # 2) Объединим train и test, чтобы получить датасет для adversarial learning
        #    Добавим колонку adv_target = 0 для train, 1 для test
        train_sdf = train_sdf.withColumn("adv_target", F.lit(0))
        test_sdf = test_sdf.withColumn("adv_target", F.lit(1))

        # 3) Объединим в один DataFrame
        advers_sdf = train_sdf.unionByName(test_sdf)

        # 4) Сэмплирование
        # Проверим, сколько всего строк в объединённом датасете
        total_cnt = advers_sdf.count()
        if self.sample_size and int(self.sample_size) > 0 and self.sample_size < total_cnt:
            fraction = self.sample_size / float(total_cnt)
            advers_sdf = advers_sdf.sample(False, fraction, seed=42)

        # 5) Добавим колонку fold_id — номера фолдов от 0 до cv_folds-1 (равномерное распределение)
        #    Также добавим колонку is_train (0/1) для train/val внутри фолда
        #    Простейший способ — сгенерировать случайные числа.
        #    Здесь для демонстрации используем uniform(0, 1) -> для fold и is_train
        #    В реальном коде лучше контролировать баланс и т.д.
        advers_sdf = (
            advers_sdf.withColumn(
                "fold_id",
                (F.rand(seed=42) * self.cv_folds).cast("int")
            )
            .withColumn(
                "is_train",
                (F.rand(seed=43) > 0.5).cast("int")
            )
        ).cache()

        # Для итеративного удаления фичей
        removal_history = []
        features_to_remove = []
        current_selected_cols = features.copy()

        previous_metric = 1.0  # Изначально ставим высокий, чтобы посчитать дельту
        prev_removed_feature = None
        start_time = time.time()
        iteration = 0

        # Основной цикл
        while True:
            print("-" * 50)
            iter_start_time = dt.datetime.now()

            # 6) Проверка времени
            elapsed_time = time.time() - start_time
            if elapsed_time > self.max_time:
                print("[AdversarialFeatureRemover] Достигнут лимит времени.")
                break

            # 7) Посчитаем метрику и feature_importances на текущем списке фичей
            #    (Усреднённую по фолдам). Для этого используем PySpark groupBy("fold_id").applyInPandas(...)

            # Если после нескольких итераций число фичей сильно уменьшится, нужно проверить, не пуст ли список
            if not current_selected_cols:
                print("[AdversarialFeatureRemover] Список фичей опустел!")
                break

            # Собираем DataFrame для вычисления
            # На каждой итерации отсекаем удалённые фичи перед подачей в train_on_fold_func
            # (внутри pandas udf всё равно нужно будет фильтровать колонки по current_selected_cols).

            # Подготовим Spark UDF-функцию (через applyInPandas) для обучения и валидации на фолде
            train_on_fold_func = self.get_train_on_fold_func(
                features=current_selected_cols,
                cat_cols=[c for c in categorical_features if c in current_selected_cols],
                num_boost_round=self.num_boost_round,
                importance_type=self.importance_type,
                learning_rate=self.learning_rate,
            )

            # Схема данных, которые вернёт train_on_fold_func
            schema = "feature_name string, importance_gain double, num_trees int, fold_id int, fold_auc: double"

            # Запускаем groupBy + applyInPandas
            importances_sdf = (
                advers_sdf.groupBy("fold_id")
                .applyInPandas(train_on_fold_func, schema=schema)
            )

            # Собираем результат в драйвер
            importances_pdf = importances_sdf.toPandas()

            # Если фолды вернули пустые результаты или у нас недостаточно фолдов
            if importances_pdf.empty:
                raise ValueError("[AdversarialFeatureRemover] Не удалось получить результат от applyInPandas.")

            # Усредняем roc_auc по фолдам
            mean_auc = importances_pdf.groupby("fold_id")["fold_auc"].first().mean()
            # Усредняем (или суммируем и потом делим) важность фичей
            # Важно, что в applyInPandas мы возвращаем одну и ту же длину по каждой фиче, поэтому
            # можно группировать по feature_name
            feature_importance_series = importances_pdf.groupby("feature_name")["importance_gain"].mean()

            mean_trees = importances_pdf.groupby("fold_id")["num_trees"].first().mean()

            print(f"[AdversarialFeatureRemover] Iteration={iteration}, mean AUC={mean_auc:.6f}")

            # 8) Проверяем критерии остановки
            if mean_auc <= self.target_metric_value:
                print(f"[AdversarialFeatureRemover] AUC={mean_auc:.6f} <= target_metric={self.target_metric_value}")
                break
            if len(features_to_remove) >= self.max_features_to_remove:
                print(f"[AdversarialFeatureRemover] Достигнут лимит удаления фичей: {self.max_features_to_remove}")
                break

            # 9) Выбираем фичу с максимальным gain, удаляем
            # Если feature_importance_series пуст, выходим
            if feature_importance_series.empty:
                print("[AdversarialFeatureRemover] Не удалось получить importances. Останавливаемся.")
                break

            top_feature = feature_importance_series.idxmax()
            importance_value = feature_importance_series.max()

            delta = previous_metric - mean_auc

            if prev_removed_feature is None:
                print(f"Initial remove => '{top_feature}', importance={importance_value:.3f}, Trees: {mean_trees:.1f}")
            else:
                print(f"Remove => '{top_feature}', importance={importance_value:.3f}, delta={delta:.6f}, Trees: {mean_trees:.1f}")

            # Удаляем фичу из current_selected_cols, если она там есть
            if top_feature in current_selected_cols:
                current_selected_cols.remove(top_feature)

            # Удаляем фичу из categorical_features, если она там была
            if top_feature in categorical_features:
                categorical_features.remove(top_feature)

            # Добавляем её в список удалённых
            features_to_remove.append(top_feature)

            # Для истории — записываем результат
            # Если мы делаем JSON-файл, сохраняем на каждой итерации
            if prev_removed_feature is not None:
                removal_history.append({
                    "feature": prev_removed_feature,
                    "delta": previous_metric - mean_auc,
                    "metric": mean_auc,
                })
                self.save_removal_history(removal_history)

            # Для лога
            if prev_removed_feature is not None:
                print(f"Remaining: {len(current_selected_cols)+1} | "
                      f"Removed: {len(features_to_remove)-1} | "
                      f"AUC: {previous_metric:.6f} => {mean_auc:.6f} "
                      f"[{str(dt.datetime.now() - iter_start_time)}]")

            # Обновляем метрику
            previous_metric = mean_auc
            prev_removed_feature = top_feature
            iteration += 1

        # Итого, при выходе из цикла, мы имеем в features_to_remove список удалённых фичей.
        # Запишем историю с последней итерацией
        if prev_removed_feature is not None:
            removal_history.append({
                "feature": prev_removed_feature,
                "delta": 0.0,  # или то, что хотим записать напоследок
                "metric": previous_metric
            })
            self.save_removal_history(removal_history)

        print(f"[AdversarialFeatureRemover] Итого удалено фичей: {len(features_to_remove)}")
        return features_to_remove

    def get_features_to_include(self, train_sdf, test_sdf=None):
        return []

    @staticmethod
    def get_train_on_fold_func(features, cat_cols, num_boost_round, importance_type, learning_rate):
        """
        Генерирует функцию, которую будет вызывать applyInPandas для обучения на фолде.

        Ожидается, что в pdf будет столбец fold_id (одинаковый в одной группе),
        а также столбец is_train (0/1).
        """
        def train_on_fold_func(pdf: pd.DataFrame) -> pd.DataFrame:
            fold_id = pdf["fold_id"].iloc[0]

            # Проверка корректности разбиения (чтобы не было константности метки)
            # Здесь можно проверять, что is_train==0/1 встречается в pdf
            train_cnt = (pdf["is_train"] == 1).sum()
            valid_cnt = (pdf["is_train"] == 0).sum()
            total_cnt = len(pdf)
            if train_cnt == 0 or valid_cnt == 0:
                raise ValueError(
                    f"[train_on_fold_func] fold_id={fold_id} содержит "
                    f"train_cnt={train_cnt}, valid_cnt={valid_cnt}. Невозможно обучить модель."
                )
            # Также можно проверять долю, если нужно
            if train_cnt < 0.2 * total_cnt or valid_cnt < 0.2 * total_cnt:
                raise ValueError(
                    f"[train_on_fold_func] fold_id={fold_id} соотношение train/valid < 20%. "
                    f"train_cnt={train_cnt}, valid_cnt={valid_cnt}, total={total_cnt}"
                )

            # Разделяем на train/val
            train_pdf = pdf[pdf["is_train"] == 1]
            valid_pdf = pdf[pdf["is_train"] == 0]

            # Формируем матрицы
            X_train = train_pdf[features]
            y_train = train_pdf["adv_target"]
            X_valid = valid_pdf[features]
            y_valid = valid_pdf["adv_target"]

            # Создаём lgbm.Dataset
            train_dataset = lgbm.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
            valid_dataset = lgbm.Dataset(X_valid, label=y_valid, reference=train_dataset, categorical_feature=cat_cols)

            lgbm_params = {
                'objective': "binary",
                'metric': "auc",
                'boosting': "gbdt",
                'verbosity': -1,
                'seed': 42,
                'learning_rate': learning_rate,
                'num_threads': 8,
                'max_depth': 16,
                'num_leaves': 244,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.9,
                'bagging_freq': 1,
                'max_bin': 63,
                'early_stopping_rounds': 25,
            }

            # Обучаем модель
            model = lgbm.train(
                params=lgbm_params,
                train_set=train_dataset,
                num_boost_round=num_boost_round,
                valid_sets=[valid_dataset],
            )

            # Предикт на валидацию
            oof_preds = model.predict(X_valid)
            # Считаем ROC-AUC
            fold_auc = roc_auc_score(y_valid, oof_preds)

            trees = model.num_trees()

            # Достаём feature importances
            if importance_type == 'shap':
                shap_values_full = model.predict(X_valid, pred_contrib=True)
                shap_values = shap_values_full[:, :-1]
                importances = np.mean(np.abs(shap_values), axis=0)
                feat_names = model.feature_name()
            else:
                importances = model.feature_importance(importance_type=importance_type)
                feat_names = model.feature_name()

            # Готовим DF для возврата
            # На каждой фиче повторяем fold_auc, чтобы потом агрегировать
            result_df = pd.DataFrame({
                "feature_name": feat_names,
                "importance_gain": importances,
                "num_trees": trees,
                "fold_id": [fold_id] * len(feat_names),
                "fold_auc": [fold_auc] * len(feat_names)
            })
            return result_df

        # Не забывайте импортировать roc_auc_score:
        from sklearn.metrics import roc_auc_score

        return train_on_fold_func

    def save_removal_history(self, removal_history: list):
        """
        Сохраняет историю удаления в JSON-файл (removed_feature_file).
        Вызывается после каждой итерации.
        """
        with open(self.removed_feature_file, 'w', encoding='utf-8') as f:
            json.dump(removal_history, f, indent=4, ensure_ascii=False)
        # print(f"[AdversarialFeatureRemover] Записана история удаления в {self.removed_feature_file}")
