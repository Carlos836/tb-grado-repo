"""
Modelos específicos para scoring crediticio.
Implementa XGBoost, CatBoost, LightGBM, HistGradientBoosting, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from .base_model import BaseCreditScoringModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseCreditScoringModel):
    """
    Modelo XGBoost para scoring crediticio.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "XGBoost")
        
        # Configuración específica de XGBoost
        self.xgb_config = config.get('ml_models', {}).get('default_params', {}).get('XGBoost', {})
        
        logger.info(f"XGBoost configurado")
    
    def _create_model(self, **params) -> Any:
        """Crea instancia de XGBoost."""
        try:
            import xgboost as xgb
            
            # Parámetros por defecto
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'eval_metric': 'auc',
                'use_label_encoder': False
            }
            
            # Actualizar con parámetros proporcionados
            default_params.update(params)
            
            return xgb.XGBClassifier(**default_params)
            
        except ImportError:
            raise ImportError("XGBoost no está instalado. Instalar con: pip install xgboost")
    
    def get_default_params(self) -> Dict[str, Any]:
        """Obtiene parámetros por defecto de XGBoost."""
        return {
            'n_estimators': self.xgb_config.get('n_estimators', 100),
            'max_depth': self.xgb_config.get('max_depth', 6),
            'learning_rate': self.xgb_config.get('learning_rate', 0.1),
            'random_state': self.xgb_config.get('random_state', 42),
            'eval_metric': 'auc',
            'use_label_encoder': False
        }


class CatBoostModel(BaseCreditScoringModel):
    """
    Modelo CatBoost para scoring crediticio.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "CatBoost")
        
        # Configuración específica de CatBoost
        self.catboost_config = config.get('ml_models', {}).get('default_params', {}).get('CatBoost', {})
        
        logger.info(f"CatBoost configurado")
    
    def _create_model(self, **params) -> Any:
        """Crea instancia de CatBoost."""
        try:
            from catboost import CatBoostClassifier
            
            # Parámetros por defecto
            default_params = {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1,
                'random_seed': 42,
                'verbose': False,
                'eval_metric': 'AUC'
            }
            
            # Actualizar con parámetros proporcionados
            default_params.update(params)
            
            return CatBoostClassifier(**default_params)
            
        except ImportError:
            raise ImportError("CatBoost no está instalado. Instalar con: pip install catboost")
    
    def get_default_params(self) -> Dict[str, Any]:
        """Obtiene parámetros por defecto de CatBoost."""
        return {
            'iterations': self.catboost_config.get('iterations', 100),
            'depth': self.catboost_config.get('depth', 6),
            'learning_rate': self.catboost_config.get('learning_rate', 0.1),
            'random_seed': self.catboost_config.get('random_seed', 42),
            'verbose': False,
            'eval_metric': 'AUC'
        }


class LightGBMModel(BaseCreditScoringModel):
    """
    Modelo LightGBM para scoring crediticio.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "LightGBM")
        
        # Configuración específica de LightGBM
        self.lgb_config = config.get('ml_models', {}).get('default_params', {}).get('LightGBM', {})
        
        logger.info(f"LightGBM configurado")
    
    def _create_model(self, **params) -> Any:
        """Crea instancia de LightGBM."""
        try:
            import lightgbm as lgb
            
            # Parámetros por defecto
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbose': -1,
                'metric': 'auc'
            }
            
            # Actualizar con parámetros proporcionados
            default_params.update(params)
            
            return lgb.LGBMClassifier(**default_params)
            
        except ImportError:
            raise ImportError("LightGBM no está instalado. Instalar con: pip install lightgbm")
    
    def get_default_params(self) -> Dict[str, Any]:
        """Obtiene parámetros por defecto de LightGBM."""
        return {
            'n_estimators': self.lgb_config.get('n_estimators', 100),
            'max_depth': self.lgb_config.get('max_depth', 6),
            'learning_rate': self.lgb_config.get('learning_rate', 0.1),
            'random_state': self.lgb_config.get('random_state', 42),
            'verbose': -1,
            'metric': 'auc'
        }


class HistGradientBoostingModel(BaseCreditScoringModel):
    """
    Modelo HistGradientBoosting para scoring crediticio.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "HistGradientBoosting")
        
        logger.info(f"HistGradientBoosting configurado")
    
    def _create_model(self, **params) -> Any:
        """Crea instancia de HistGradientBoosting."""
        from sklearn.ensemble import HistGradientBoostingClassifier
        
        # Parámetros por defecto
        default_params = {
            'max_iter': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'scoring': 'roc_auc'
        }
        
        # Actualizar con parámetros proporcionados
        default_params.update(params)
        
        return HistGradientBoostingClassifier(**default_params)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Obtiene parámetros por defecto de HistGradientBoosting."""
        return {
            'max_iter': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'scoring': 'roc_auc'
        }


class RandomForestModel(BaseCreditScoringModel):
    """
    Modelo Random Forest para scoring crediticio.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "RandomForest")
        
        logger.info(f"RandomForest configurado")
    
    def _create_model(self, **params) -> Any:
        """Crea instancia de Random Forest."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Parámetros por defecto
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Actualizar con parámetros proporcionados
        default_params.update(params)
        
        return RandomForestClassifier(**default_params)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Obtiene parámetros por defecto de Random Forest."""
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        }


class LogisticRegressionModel(BaseCreditScoringModel):
    """
    Modelo Logistic Regression para scoring crediticio.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "LogisticRegression")
        
        logger.info(f"LogisticRegression configurado")
    
    def _create_model(self, **params) -> Any:
        """Crea instancia de Logistic Regression."""
        from sklearn.linear_model import LogisticRegression
        
        # Parámetros por defecto
        default_params = {
            'random_state': 42,
            'max_iter': 1000,
            'solver': 'liblinear'
        }
        
        # Actualizar con parámetros proporcionados
        default_params.update(params)
        
        return LogisticRegression(**default_params)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Obtiene parámetros por defecto de Logistic Regression."""
        return {
            'random_state': 42,
            'max_iter': 1000,
            'solver': 'liblinear'
        }


class CreditScoringModels:
    """
    Factory para modelos de scoring crediticio.
    """
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> BaseCreditScoringModel:
        """
        Crea un modelo específico.
        
        Args:
            model_type: Tipo de modelo ('XGBoost', 'CatBoost', etc.)
            config: Configuración del modelo
            
        Returns:
            Instancia del modelo
        """
        models = {
            'XGBoost': XGBoostModel,
            'CatBoost': CatBoostModel,
            'LightGBM': LightGBMModel,
            'HistGradientBoosting': HistGradientBoostingModel,
            'RandomForest': RandomForestModel,
            'LogisticRegression': LogisticRegressionModel,
        }
        
        if model_type not in models:
            raise ValueError(f"Modelo no soportado: {model_type}")
        
        return models[model_type](config)
    
    @staticmethod
    def get_available_models() -> list:
        """Obtiene lista de modelos disponibles."""
        return ['XGBoost', 'CatBoost', 'LightGBM', 'HistGradientBoosting', 'RandomForest', 'LogisticRegression']
    
    @staticmethod
    def get_model_info(model_type: str) -> Dict[str, Any]:
        """Obtiene información sobre un modelo específico."""
        info = {
            'XGBoost': {
                'description': 'Extreme Gradient Boosting',
                'library': 'xgboost',
                'type': 'Gradient Boosting',
                'best_for': 'Datos tabulares con alta dimensionalidad'
            },
            'CatBoost': {
                'description': 'Categorical Boosting',
                'library': 'catboost',
                'type': 'Gradient Boosting',
                'best_for': 'Datos con variables categóricas'
            },
            'LightGBM': {
                'description': 'Light Gradient Boosting Machine',
                'library': 'lightgbm',
                'type': 'Gradient Boosting',
                'best_for': 'Datos grandes con alta velocidad'
            },
            'HistGradientBoosting': {
                'description': 'Histogram-based Gradient Boosting',
                'library': 'sklearn',
                'type': 'Gradient Boosting',
                'best_for': 'Datos tabulares con sklearn'
            },
            'RandomForest': {
                'description': 'Random Forest Classifier',
                'library': 'sklearn',
                'type': 'Ensemble',
                'best_for': 'Datos con overfitting'
            },
            'LogisticRegression': {
                'description': 'Logistic Regression',
                'library': 'sklearn',
                'type': 'Linear',
                'best_for': 'Baseline y interpretabilidad'
            }
        }
        
        return info.get(model_type, {})
