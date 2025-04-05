"""
MLOps CSY Library
~~~~~~~~~~~~~~~~

A library for machine learning operations with ROC-AUC evaluation.
"""

from .metrics import calculate_roc_auc, plot_roc_curve
from .model_utils import train_model, predict
from .loader import load_data
from .explorer import explore_data
from .visualizer import visualize_data

__version__ = "0.1.0"

__all__ = [
    'calculate_roc_auc',
    'plot_roc_curve',
    'train_model',
    'predict',
    'load_data',
    'explore_data',
    'visualize_data'
] 