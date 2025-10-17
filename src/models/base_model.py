"""
Modelo base para scoring crediticio.
Define la interfaz común para todos los modelos de ML.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class BaseCreditScoringModel(ABC):
    """
    Clase base abstracta para todos los modelos de scoring crediticio.
    """
    
    def __init__(self, config: Dict[str, Any], model_name: str):
        """
        Inicializa el modelo base.
        
        Args:
            config: Configuración del modelo
            model_name: Nombre del modelo
        """
        self.config = config
        self.model_name = model_name
        self.is_fitted = False
        self.model = None
        self.feature_names = None
        self.target_name = None
        
        # Configuración específica del modelo
        self.ml_config = config.get('ml_models', {})
        self.training_config = self.ml_config.get('training', {})
        self.cv_folds = self.training_config.get('cv_folds', 5)
        self.random_state = self.training_config.get('random_state', 42)
        self.test_size = self.training_config.get('test_size', 0.2)
        
        # Métricas de evaluación
        self.evaluation_metrics = config.get('evaluation', {}).get('model_performance', [])
        self.credit_metrics = config.get('evaluation', {}).get('credit_metrics', [])
        
        logger.info(f"Modelo {model_name} inicializado")
    
    @abstractmethod
    def _create_model(self, **params) -> Any:
        """
        Crea la instancia del modelo específico.
        
        Args:
            **params: Parámetros del modelo
            
        Returns:
            Instancia del modelo
        """
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """
        Obtiene parámetros por defecto del modelo.
        
        Returns:
            Diccionario con parámetros por defecto
        """
        pass
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> None:
        """
        Entrena el modelo.
        
        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento
            **fit_params: Parámetros adicionales de entrenamiento
        """
        if not self.validate_data(X, y):
            raise ValueError("Datos no válidos para entrenamiento")
        
        try:
            logger.info(f"Entrenando modelo {self.model_name}")
            
            # Obtener parámetros por defecto
            default_params = self.get_default_params()
            default_params.update(fit_params)
            
            # Crear modelo
            self.model = self._create_model(**default_params)
            
            # Guardar nombres de features y target
            self.feature_names = X.columns.tolist()
            self.target_name = y.name if hasattr(y, 'name') else 'target'
            
            # Entrenar modelo
            self.model.fit(X, y)
            self.is_fitted = True
            
            logger.info(f"Modelo {self.model_name} entrenado exitosamente")
            
        except Exception as e:
            logger.error(f"Error entrenando modelo {self.model_name}: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones.
        
        Args:
            X: Features para predicción
            
        Returns:
            Predicciones
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe estar entrenado antes de hacer predicciones")
        
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error haciendo predicciones con {self.model_name}: {str(e)}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones de probabilidad.
        
        Args:
            X: Features para predicción
            
        Returns:
            Probabilidades de predicción
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe estar entrenado antes de hacer predicciones")
        
        try:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                # Para modelos que no tienen predict_proba, usar decision_function
                if hasattr(self.model, 'decision_function'):
                    scores = self.model.decision_function(X)
                    # Convertir scores a probabilidades usando sigmoid
                    probabilities = 1 / (1 + np.exp(-scores))
                    return np.column_stack([1 - probabilities, probabilities])
                else:
                    raise ValueError(f"Modelo {self.model_name} no soporta predicciones de probabilidad")
        except Exception as e:
            logger.error(f"Error haciendo predicciones de probabilidad con {self.model_name}: {str(e)}")
            raise
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      scoring: str = 'roc_auc', cv: Optional[int] = None) -> Dict[str, Any]:
        """
        Realiza validación cruzada.
        
        Args:
            X: Features
            y: Target
            scoring: Métrica de evaluación
            cv: Número de folds (opcional)
            
        Returns:
            Resultados de validación cruzada
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe estar entrenado antes de validación cruzada")
        
        if cv is None:
            cv = self.cv_folds
        
        try:
            logger.info(f"Realizando validación cruzada con {cv} folds")
            
            # Crear modelo temporal para CV
            temp_model = self._create_model(**self.get_default_params())
            
            # Validación cruzada
            cv_scores = cross_val_score(
                temp_model, X, y, 
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
                scoring=scoring
            )
            
            results = {
                'cv_scores': cv_scores,
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'scoring': scoring,
                'cv_folds': cv
            }
            
            logger.info(f"CV completado: {scoring} = {results['mean_score']:.4f} ± {results['std_score']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error en validación cruzada: {str(e)}")
            raise
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Evalúa el modelo con múltiples métricas.
        
        Args:
            X: Features de prueba
            y: Target de prueba
            
        Returns:
            Diccionario con métricas de evaluación
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe estar entrenado antes de evaluación")
        
        try:
            logger.info(f"Evaluando modelo {self.model_name}")
            
            # Predicciones
            y_pred = self.predict(X)
            y_proba = self.predict_proba(X)
            
            # Si y_proba tiene 2 columnas, tomar la segunda (clase positiva)
            if y_proba.shape[1] == 2:
                y_proba_positive = y_proba[:, 1]
            else:
                y_proba_positive = y_proba
            
            # Métricas básicas
            metrics = {}
            
            # AUC-ROC
            if 'auc_roc' in self.evaluation_metrics:
                try:
                    metrics['auc_roc'] = roc_auc_score(y, y_proba_positive)
                except Exception as e:
                    logger.warning(f"Error calculando AUC-ROC: {e}")
                    metrics['auc_roc'] = 0.0
            
            # Precision, Recall, F1
            if 'precision' in self.evaluation_metrics:
                metrics['precision'] = precision_score(y, y_pred, average='weighted')
            
            if 'recall' in self.evaluation_metrics:
                metrics['recall'] = recall_score(y, y_pred, average='weighted')
            
            if 'f1_score' in self.evaluation_metrics:
                metrics['f1_score'] = f1_score(y, y_pred, average='weighted')
            
            # Métricas específicas de crédito
            if 'gini_coefficient' in self.credit_metrics:
                try:
                    auc = metrics.get('auc_roc', 0.0)
                    metrics['gini_coefficient'] = 2 * auc - 1
                except Exception as e:
                    logger.warning(f"Error calculando Gini: {e}")
                    metrics['gini_coefficient'] = 0.0
            
            logger.info(f"Evaluación completada para {self.model_name}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluando modelo {self.model_name}: {str(e)}")
            raise
    
    def validate_data(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """
        Valida que los datos sean adecuados para el modelo.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            True si los datos son válidos
        """
        if X is None or X.empty:
            logger.error("Features vacías o nulas")
            return False
        
        if y is None or y.empty:
            logger.error("Target vacío o nulo")
            return False
        
        if len(X) != len(y):
            logger.error(f"Longitud de features ({len(X)}) no coincide con target ({len(y)})")
            return False
        
        if len(X.columns) == 0:
            logger.error("Features sin columnas")
            return False
        
        logger.info(f"Datos validados: {X.shape[0]} filas, {X.shape[1]} features")
        return True
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Obtiene importancia de features.
        
        Returns:
            Serie con importancia de features
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe estar entrenado antes de obtener importancia de features")
        
        if self.feature_names is None:
            raise ValueError("Nombres de features no disponibles")
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
            elif hasattr(self.model, 'coef_'):
                # Para modelos lineales, usar coeficientes
                coef = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
                return pd.Series(np.abs(coef), index=self.feature_names).sort_values(ascending=False)
            else:
                logger.warning(f"Modelo {self.model_name} no soporta importancia de features")
                return None
                
        except Exception as e:
            logger.error(f"Error obteniendo importancia de features: {str(e)}")
            return None
    
    def save_model(self, file_path: str) -> None:
        """
        Guarda el modelo entrenado.
        
        Args:
            file_path: Ruta donde guardar
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe estar entrenado antes de guardarlo")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'model_name': self.model_name,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, file_path)
        logger.info(f"Modelo guardado en: {file_path}")
    
    def load_model(self, file_path: str) -> None:
        """
        Carga un modelo entrenado.
        
        Args:
            file_path: Ruta del archivo
        """
        model_data = joblib.load(file_path)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.target_name = model_data['target_name']
        self.model_name = model_data['model_name']
        self.config = model_data['config']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Modelo cargado desde: {file_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información del modelo.
        
        Returns:
            Diccionario con información del modelo
        """
        return {
            'name': self.model_name,
            'type': 'CreditScoring',
            'is_fitted': self.is_fitted,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'target_name': self.target_name,
            'supports_probability': hasattr(self.model, 'predict_proba') if self.model else False,
            'supports_feature_importance': hasattr(self.model, 'feature_importances_') if self.model else False
        }
    
    def __str__(self) -> str:
        """Representación string del modelo."""
        return f"{self.model_name}(fitted={self.is_fitted})"
    
    def __repr__(self) -> str:
        """Representación detallada del modelo."""
        return f"{self.__class__.__name__}(name='{self.model_name}', fitted={self.is_fitted})"
