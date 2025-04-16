"""
Библиотека для предобработки данных и отбора фичей.
"""

from .feature_selector import FeatureSelector
from .corr_feature_remover import CorrFeatureRemover
from .preprocessing import PreprocessingPipeline

__all__ = [
    "FeatureSelector",
    "CorrFeatureRemover",
    "PreprocessingPipeline",
]
