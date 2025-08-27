from abc import ABC, abstractmethod

class BaseFeatureSelector(ABC):
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