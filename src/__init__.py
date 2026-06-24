# src/__init__.py
"""
Módulo de código fuente para el proyecto Titanic.

Este módulo contiene las clases principales para:
- Carga de datos (TitanicDataLoader)
- Ingeniería de características (FeatureEngineer)
- Preprocesamiento (TitanicPreprocessor)
- Entrenamiento de modelos (TitanicModelTrainer)
- Predicción (TitanicPredictor)
"""

from src.data_loader import TitanicDataLoader
from src.features import FeatureEngineer
from src.preprocessor import TitanicPreprocessor
from src.model_trainer import TitanicModelTrainer
from src.predict import TitanicPredictor

__all__ = [
    'TitanicDataLoader',
    'FeatureEngineer',
    'TitanicPreprocessor',
    'TitanicModelTrainer',
    'TitanicPredictor',
]

__version__ = '1.0.0'