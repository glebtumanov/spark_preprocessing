import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
import lightgbm as lgbm
from sklearn.metrics import roc_auc_score, mean_squared_error

from .base_feature_selector import BaseFeatureSelector


def get_task_config(task_type):
    """Возвращает конфигурацию LightGBM и метрики в зависимости от типа задачи"""
    if task_type == "binary_classification":
        return {
            'objective': 'binary',
            'metric': 'auc',
            'eval_metric': roc_auc_score,
        }
    elif task_type == "regression":
        return {
            'objective': 'regression',
            'metric': 'rmse', 
            'eval_metric': lambda y_true, y_pred: -mean_squared_error(y_true, y_pred, squared=False),  # Используем негативный RMSE, поэтому больше = лучше
        }
    else:
        raise ValueError(f"Неподдерживаемый тип задачи: {task_type}")


class ForwardFeatureSelector(BaseFeatureSelector):
    """
    Класс для forward selection на PySpark.
    Алгоритм:
      1. На основе всего набора фич вычисляются начальные importances через кросс-валидацию (используя applyInPandas).
      2. Фичи сортируются по убыванию важности.
      3. Итеративно по порядку каждая фича добавляется к уже выбранным, затем для полученного набора вычисляется
         метрика (ROC-AUC) на кросс-валидации (определяемой через fold_id и is_train).
         Если метрика улучшилась (либо, при votes_only, фича доминирует хотя бы в q фолдах), то фича остаётся, иначе —
         откатывается.
      4. Если подряд tol фич не дают улучшения, итерация прекращается.
    """

    def __init__(self, config: dict, sc):
        """
        config: dict — общий конфиг пайплайна (ожидается наличие секции forward_selection)
        sc: SparkContext
        """
        self.config = config
        self.sc = sc
        self.fs_config = self.config["preprocessing"].get("forward_selection")

        # Параметры для вычисления initial importances
        self.cv_folds_importances = self.fs_config.get("cv_folds_importances")
        self.num_boost_round_importances = self.fs_config.get("num_boost_round_importances")
        self.seed = self.fs_config.get("seed", 0)
        self.importance_type = self.fs_config.get("importance_type", "gain")
        self.sample_size = self.fs_config.get("sample_size", None)
        self.learning_rate = self.fs_config.get("learning_rate", 0.05)
        self.verbose = True

        # Параметры для оценки модели в процессе отбора фич
        self.cv_folds_selection = self.fs_config.get("cv_folds_selection", 5)
        self.num_boost_round_selection = self.fs_config.get("num_boost_round_selection", 300)

        # Параметры forward selection
        self.eps = self.fs_config.get("eps", 1e-4)
        self.q = self.fs_config.get("q", 2)
        self.tol = self.fs_config.get("tol", 10)
        self.votes_only = self.fs_config.get("votes_only", False)
        self.valid_frac = self.fs_config.get("valid_frac", 0.2)

        self.target_col = config["common"]["target_col"]
        self.task_type = config["common"].get("task_type", "binary_classification")
        
        # Получаем конфигурацию для типа задачи
        task_config = get_task_config(self.task_type)

        self.lgb_importance_params = {
            'objective': task_config['objective'],
            'metric': task_config['metric'],
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'seed': self.seed,
            'learning_rate': self.learning_rate,
            'max_depth': 63,
            'num_leaves': 500,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'reg_lambda': 0.001,
            'min_data_in_bin': 10,
            'min_child_weight': 10.,
            'min_sum_hessian_in_leaf': 0.1,
            'max_bin': 255,
            'early_stopping_rounds': 50
        }

        self.lgb_selection_params = {
            'objective': task_config['objective'],
            'metric': task_config['metric'],
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'seed': self.seed,
            'learning_rate': 0.03,
            'max_depth': 32,
            'num_leaves': 244,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.95,
            'bagging_freq': 1,
            'max_bin': 63,
            'min_sum_hessian_in_leaf': 1.0,
            'early_stopping_rounds': 100
        }
        
        # Сохраняем функцию оценки
        self.eval_metric = task_config['eval_metric']

    def get_features_to_exclude(
        self,
        train_sdf: DataFrame,
        test_sdf: DataFrame = None,
        features: list = None,
        categorical_features: list = None
    ) -> list:
        """
        Для forward selection используется список включаемых фич, поэтому исключаемых не определяем.
        """
        return []

    def get_features_to_include(
        self,
        train_sdf: DataFrame,
        test_sdf: DataFrame = None,
        features: list = None,
        categorical_features: list = None
    ) -> list:
        """
        Основной метод для отбора фич методом forward selection.
        1) Сначала вычисляются initial importances по всем фичам с помощью groupBy("fold_id").applyInPandas.
        2) Фичи сортируются по убыванию важности.
        3) Итеративно тестируется добавление каждой фичи: на каждом шаге запускается кросс-валидация
           (через applyInPandas, используя fold_id и is_train).
        4) Фича принимается, если выполняется условие (либо улучшение средней метрики, либо votes_only – число фолдов с улучшением ≥ q).
        5) Если tol подряд неудачных попыток, итерация завершается.
        """
        if not features:
            raise ValueError("[ForwardFeatureSelector] Пустой список фичей")

        total_cnt = train_sdf.count()
        if self.sample_size and int(self.sample_size) > 0 and self.sample_size < total_cnt:
            fraction = self.sample_size / float(total_cnt)
            train_sdf = train_sdf.sample(False, fraction, seed=42)

        # --- Шаг 1. Вычисляем initial importances по всем фичам ---
        importance_sdf = train_sdf.withColumn(
            "fold_id", F.floor(F.rand(seed=self.seed) * self.cv_folds_importances)
        ).withColumn(
            "is_train", (F.rand(seed=self.seed + 1) > self.valid_frac).cast("int")
        )
        imp_schema = "fold_id int, feature_name string, importance double, fold_auc double"

        importance_func = ForwardFeatureSelector.get_importance_cv_func(
            features,
            self.lgb_importance_params,
            categorical_features,
            self.num_boost_round_importances,
            self.target_col,
            self.importance_type,
            self.eval_metric
        )

        imp_result_sdf = importance_sdf.groupBy("fold_id").applyInPandas(importance_func, schema=imp_schema)
        imp_pdf = imp_result_sdf.toPandas()
        agg_imp = imp_pdf.groupby("feature_name")[["importance", "fold_auc"]].mean().sort_values('importance', ascending=False)
        mean_auc = agg_imp["fold_auc"].iloc[0]
        sorted_features = agg_imp.index.tolist()

        print(f"[ForwardFeatureSelector] Importances calculated on {self.cv_folds_importances} folds, "
              f"method: {self.importance_type}, "
              f"base auc: {mean_auc:.4f}, "
              f"{len(sorted_features)} features")

        # --- Шаг 2. Подготавливаем данные для кросс-валидации отбора ---
        # Для стабильности разбиения используем один и тот же split (fold_id, is_train)
        selection_sdf = train_sdf.withColumn(
            "fold_id", F.floor(F.rand(seed=self.seed) * self.cv_folds_selection)
        ).withColumn(
            "is_train", (F.rand(seed=self.seed + 1) > 0.3).cast("int")
        ).cache()

        # --- Шаг 3. Итеративный forward selection ---
        current_features = []
        rec_score = 0.0
        max_score = 0.0
        lag_fold_scores = np.zeros(self.cv_folds_selection)
        k = 0  # счетчик неудачных попыток

        if self.votes_only:
            print(f"[ForwardFeatureSelector] starting, total features: {len(sorted_features)}, "
                  f"cat: {len(categorical_features)}, "
                  f"based only on votes, minimum {self.q} of {self.cv_folds_selection}")
        else:
            print(f"[ForwardFeatureSelector] starting, total features: {len(sorted_features)}, "
                  f"cat: {len(categorical_features)}, based on votes and auc, "
                  f"minimum {self.q} of {self.cv_folds_selection}, "
                  f"min_emprovement: {self.eps:g}")

        for idx, feat in enumerate(sorted_features):
            if self.verbose:
                print('-' * 60)
                print(f"[Checking feature: {idx+1} of {len(sorted_features)}], selected: {len(current_features)}, name: {feat}")
            candidate_features = current_features + [feat]
            exists_cat_cols = [cat for cat in categorical_features if cat in candidate_features]

            cv_func = ForwardFeatureSelector.get_selection_cv_func(
                candidate_features,
                self.lgb_selection_params,
                exists_cat_cols,
                self.num_boost_round_selection,
                self.target_col,
                self.eval_metric
            )
            cv_schema = "fold_id int, fold_auc double"
            cv_result_sdf = selection_sdf.groupBy("fold_id").applyInPandas(cv_func, schema=cv_schema)
            cv_result_pdf = cv_result_sdf.toPandas()

            # Собираем значения метрики по фолдам (ожидается, что fold_id от 0 до cv_folds_selection-1)
            fold_scores = np.zeros(self.cv_folds_selection)
            for i in range(self.cv_folds_selection):
                row = cv_result_pdf[cv_result_pdf["fold_id"] == i]
                if not row.empty:
                    fold_scores[i] = row["fold_auc"].values[0]
                else:
                    fold_scores[i] = 0.0

            mean_score = fold_scores.mean()
            n_best = np.sum(fold_scores > lag_fold_scores)

            if mean_score > max_score:
                max_score = mean_score

            if self.verbose:
                print(f"==> feature {feat}, mean AUC: {mean_score:.6f}, best in {n_best} folds")

            # Если включен votes_only – проверяем только число фолдов с улучшением;
            # иначе – ещё и улучшение средней метрики (с учетом eps)
            if self.votes_only:
                improvement = (n_best >= self.q) and (mean_score + self.eps >= rec_score)
            else:
                improvement = (n_best >= self.q) and (mean_score > rec_score + self.eps)

            if improvement:
                if self.verbose:
                    print(f"==> improvement, best folds: {n_best}/{self.cv_folds_selection}, from score: {rec_score:.6f}, delta: {mean_score-rec_score:.5f}")
                current_features = candidate_features
                rec_score = mean_score
                lag_fold_scores = fold_scores.copy()
                k = 0
            else:
                k += 1
                if self.verbose:
                    print(f"==> NOT improvement {k}/{self.tol}, best folds: {n_best}/{self.cv_folds_selection}, from score: {rec_score:.6f}, delta: {mean_score-rec_score:.5f}")
                if k >= self.tol:
                    print(f"[ForwardFeatureSelector] Прекращение итераций после {k} подряд без улучшения")
                    break

        print(f"[ForwardFeatureSelector] total selected features: {len(current_features)}, "
              f"last improvement metric: {rec_score:.4f}, max metric: {max_score:.4f}")
        return current_features

    @staticmethod
    def get_importance_cv_func(features, lgb_params, cat_cols, num_boost_round, target_col, importance_type, eval_metric):
        """
        Статический метод, возвращающий функцию для groupBy("fold_id").applyInPandas,
        которая обучает LightGBM на подвыборке (is_train==1) и вычисляет ROC-AUC и feature importance.
        """
        def importance_cv_func(pdf: pd.DataFrame) -> pd.DataFrame:
            fold_id = int(pdf["fold_id"].iloc[0])
            train_pdf = pdf[pdf["is_train"] == 1]
            valid_pdf = pdf[pdf["is_train"] == 0]
            if train_pdf.empty or valid_pdf.empty:
                raise ValueError(f"[importance_cv_func] Недостаточно данных для фолда {fold_id}")
            X_train = train_pdf[features]
            y_train = train_pdf[target_col]
            X_valid = valid_pdf[features]
            y_valid = valid_pdf[target_col]

            dtrain = lgbm.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
            dvalid = lgbm.Dataset(X_valid, label=y_valid, reference=dtrain, categorical_feature=cat_cols)

            model = lgbm.train(
                lgb_params,
                dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dvalid],
            )

            preds = model.predict(X_valid)
            fold_auc = eval_metric(y_valid, preds)

            if importance_type == 'shap':
                shap_values_full = model.predict(X_valid, pred_contrib=True)
                shap_values = shap_values_full[:, :-1]
                importances = np.mean(np.abs(shap_values), axis=0)
                feat_names = model.feature_name()
            else:
                importances = model.feature_importance(importance_type=importance_type)
                feat_names = model.feature_name()

            df_imp = pd.DataFrame({
                "fold_id": [fold_id] * len(feat_names),
                "feature_name": feat_names,
                "importance": importances,
                "fold_auc": [fold_auc] * len(feat_names)
            })

            return df_imp

        return importance_cv_func

    @staticmethod
    def get_selection_cv_func(features, lgb_params, cat_cols, num_boost_round, target_col, eval_metric):
        """
        Статический метод, возвращающий функцию для groupBy("fold_id").applyInPandas,
        которая обучает модель на подвыборке (is_train==1) и возвращает ROC-AUC на валидации.
        """
        def selection_cv_func(pdf: pd.DataFrame) -> pd.DataFrame:
            fold_id = int(pdf["fold_id"].iloc[0])
            train_pdf = pdf[pdf["is_train"] == 1]
            valid_pdf = pdf[pdf["is_train"] == 0]
            if train_pdf.empty or valid_pdf.empty:
                raise ValueError(f"[selection_cv_func] Недостаточно данных для фолда {fold_id}")
            X_train = train_pdf[features]
            y_train = train_pdf[target_col]
            X_valid = valid_pdf[features]
            y_valid = valid_pdf[target_col]

            dtrain = lgbm.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
            dvalid = lgbm.Dataset(X_valid, label=y_valid, reference=dtrain, categorical_feature=cat_cols)

            model = lgbm.train(
                lgb_params,
                dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dvalid],
            )
            preds = model.predict(X_valid)
            fold_auc = eval_metric(y_valid, preds)
            result_df = pd.DataFrame({
                "fold_id": [fold_id],
                "fold_auc": [fold_auc]
            })
            return result_df

        return selection_cv_func
