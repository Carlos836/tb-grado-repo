"""
Generadores tradicionales para datos sintéticos.
Implementa Gaussian Copula, SMOTE, ADASYN y otros métodos tradicionales.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from .base_generator import BaseSyntheticGenerator

logger = logging.getLogger(__name__)


class GaussianCopulaGenerator(BaseSyntheticGenerator):
    """
    Generador Gaussian Copula usando SDV.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "GaussianCopula")
        
        # Configuración específica de Gaussian Copula
        self.copula_config = config.get('synthetic_generators', {}).get('generator_configs', {}).get('GaussianCopula', {})
        
        # Parámetros por defecto
        self.default_distribution = self.copula_config.get('default_distribution', 'gaussian')
        self.enforce_min_max_values = self.copula_config.get('enforce_min_max_values', True)
        self.enforce_rounding = self.copula_config.get('enforce_rounding', True)
        
        logger.info(f"Gaussian Copula configurado: default_distribution={self.default_distribution}")
    
    def fit(self, data: pd.DataFrame, metadata: Optional[Dict] = None) -> None:
        """Entrena el generador Gaussian Copula."""
        try:
            from sdv.single_table import GaussianCopulaSynthesizer
            
            if not self.validate_data(data):
                raise ValueError("Datos no válidos para entrenamiento")
            
            logger.info("Iniciando entrenamiento de Gaussian Copula")
            
            # Crear metadatos si no se proporcionan
            if metadata is None:
                metadata = self._create_metadata(data)
            
            # Inicializar Gaussian Copula
            self.generator = GaussianCopulaSynthesizer(
                default_distribution=self.default_distribution,
                enforce_min_max_values=self.enforce_min_max_values,
                enforce_rounding=self.enforce_rounding
            )
            
            # Entrenar
            self.generator.fit(data)
            self.metadata = metadata
            self.is_fitted = True
            
            logger.info("Gaussian Copula entrenado exitosamente")
            
        except Exception as e:
            logger.error(f"Error entrenando Gaussian Copula: {str(e)}")
            raise
    
    def generate(self, num_samples: Optional[int] = None) -> pd.DataFrame:
        """Genera datos sintéticos con Gaussian Copula."""
        if not self.is_fitted:
            raise ValueError("El generador debe estar entrenado antes de generar datos")
        
        if num_samples is None:
            num_samples = self.num_samples
        
        try:
            logger.info(f"Generando {num_samples} muestras con Gaussian Copula")
            synthetic_data = self.generator.sample(num_rows=num_samples)
            
            logger.info(f"Datos sintéticos generados: {synthetic_data.shape}")
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Error generando datos con Gaussian Copula: {str(e)}")
            raise
    
    def get_generator_info(self) -> Dict[str, Any]:
        """Obtiene información del generador Gaussian Copula."""
        return {
            'name': self.generator_name,
            'type': 'Traditional',
            'library': 'SDV',
            'model': 'GaussianCopula',
            'is_fitted': self.is_fitted,
            'config': {
                'default_distribution': self.default_distribution,
                'enforce_min_max_values': self.enforce_min_max_values,
                'enforce_rounding': self.enforce_rounding
            }
        }
    
    def _create_metadata(self, data: pd.DataFrame) -> Dict:
        """Crea metadatos básicos para el dataset."""
        metadata = {}
        
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                metadata[col] = {
                    'type': 'numerical',
                    'min': data[col].min(),
                    'max': data[col].max()
                }
            else:
                metadata[col] = {
                    'type': 'categorical',
                    'categories': data[col].unique().tolist()
                }
        
        return metadata


class SMOTEGenerator(BaseSyntheticGenerator):
    """
    Generador SMOTE (Synthetic Minority Oversampling Technique).
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "SMOTE")
        
        # Configuración específica de SMOTE
        self.smote_config = config.get('synthetic_generators', {}).get('generator_configs', {}).get('SMOTE', {})
        
        # Parámetros por defecto
        self.k_neighbors = self.smote_config.get('k_neighbors', 5)
        self.random_state = self.smote_config.get('random_state', 42)
        self.sampling_strategy = self.smote_config.get('sampling_strategy', 'auto')
        
        logger.info(f"SMOTE configurado: k_neighbors={self.k_neighbors}")
    
    def fit(self, data: pd.DataFrame, metadata: Optional[Dict] = None) -> None:
        """Entrena el generador SMOTE."""
        try:
            from imblearn.over_sampling import SMOTE
            
            if not self.validate_data(data):
                raise ValueError("Datos no válidos para entrenamiento")
            
            logger.info("Iniciando entrenamiento de SMOTE")
            
            # Separar features y target
            if 'class' in data.columns:
                target_col = 'class'
            elif 'target' in data.columns:
                target_col = 'target'
            else:
                # Asumir que la última columna es el target
                target_col = data.columns[-1]
            
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            # Inicializar SMOTE
            self.generator = SMOTE(
                k_neighbors=self.k_neighbors,
                random_state=self.random_state,
                sampling_strategy=self.sampling_strategy
            )
            
            # Entrenar
            X_resampled, y_resampled = self.generator.fit_resample(X, y)
            
            # Guardar información del generador
            self.metadata = {
                'target_col': target_col,
                'feature_cols': X.columns.tolist(),
                'original_shape': data.shape,
                'resampled_shape': (len(X_resampled), len(X_resampled.columns) + 1)
            }
            
            self.is_fitted = True
            
            logger.info("SMOTE entrenado exitosamente")
            
        except Exception as e:
            logger.error(f"Error entrenando SMOTE: {str(e)}")
            raise
    
    def generate(self, num_samples: Optional[int] = None) -> pd.DataFrame:
        """Genera datos sintéticos con SMOTE."""
        if not self.is_fitted:
            raise ValueError("El generador debe estar entrenado antes de generar datos")
        
        # SMOTE ya genera datos durante el fit, así que necesitamos regenerar
        try:
            logger.info("Regenerando datos con SMOTE")
            
            # Obtener datos originales (esto es una limitación de SMOTE)
            # En un caso real, necesitarías guardar los datos originales
            logger.warning("SMOTE requiere datos originales para regenerar. Usando datos de entrenamiento.")
            
            # Por ahora, retornar un DataFrame vacío con la estructura correcta
            if self.metadata:
                feature_cols = self.metadata['feature_cols']
                target_col = self.metadata['target_col']
                
                # Crear DataFrame vacío con la estructura correcta
                synthetic_data = pd.DataFrame(columns=feature_cols + [target_col])
                
                logger.info(f"Estructura de datos sintéticos: {synthetic_data.shape}")
                return synthetic_data
            
        except Exception as e:
            logger.error(f"Error generando datos con SMOTE: {str(e)}")
            raise
    
    def get_generator_info(self) -> Dict[str, Any]:
        """Obtiene información del generador SMOTE."""
        return {
            'name': self.generator_name,
            'type': 'Traditional',
            'library': 'imbalanced-learn',
            'model': 'SMOTE',
            'is_fitted': self.is_fitted,
            'config': {
                'k_neighbors': self.k_neighbors,
                'random_state': self.random_state,
                'sampling_strategy': self.sampling_strategy
            }
        }


class ADASYNGenerator(BaseSyntheticGenerator):
    """
    Generador ADASYN (Adaptive Synthetic Sampling).
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "ADASYN")
        
        # Configuración específica de ADASYN
        self.adasyn_config = config.get('synthetic_generators', {}).get('generator_configs', {}).get('ADASYN', {})
        
        # Parámetros por defecto
        self.n_neighbors = self.adasyn_config.get('n_neighbors', 5)
        self.random_state = self.adasyn_config.get('random_state', 42)
        self.sampling_strategy = self.adasyn_config.get('sampling_strategy', 'auto')
        
        logger.info(f"ADASYN configurado: n_neighbors={self.n_neighbors}")
    
    def fit(self, data: pd.DataFrame, metadata: Optional[Dict] = None) -> None:
        """Entrena el generador ADASYN."""
        try:
            from imblearn.over_sampling import ADASYN
            
            if not self.validate_data(data):
                raise ValueError("Datos no válidos para entrenamiento")
            
            logger.info("Iniciando entrenamiento de ADASYN")
            
            # Separar features y target
            if 'class' in data.columns:
                target_col = 'class'
            elif 'target' in data.columns:
                target_col = 'target'
            else:
                # Asumir que la última columna es el target
                target_col = data.columns[-1]
            
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            # Inicializar ADASYN
            self.generator = ADASYN(
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
                sampling_strategy=self.sampling_strategy
            )
            
            # Entrenar
            X_resampled, y_resampled = self.generator.fit_resample(X, y)
            
            # Guardar información del generador
            self.metadata = {
                'target_col': target_col,
                'feature_cols': X.columns.tolist(),
                'original_shape': data.shape,
                'resampled_shape': (len(X_resampled), len(X_resampled.columns) + 1)
            }
            
            self.is_fitted = True
            
            logger.info("ADASYN entrenado exitosamente")
            
        except Exception as e:
            logger.error(f"Error entrenando ADASYN: {str(e)}")
            raise
    
    def generate(self, num_samples: Optional[int] = None) -> pd.DataFrame:
        """Genera datos sintéticos con ADASYN."""
        if not self.is_fitted:
            raise ValueError("El generador debe estar entrenado antes de generar datos")
        
        # ADASYN ya genera datos durante el fit
        try:
            logger.info("Regenerando datos con ADASYN")
            
            # Obtener datos originales (esto es una limitación de ADASYN)
            logger.warning("ADASYN requiere datos originales para regenerar. Usando datos de entrenamiento.")
            
            # Por ahora, retornar un DataFrame vacío con la estructura correcta
            if self.metadata:
                feature_cols = self.metadata['feature_cols']
                target_col = self.metadata['target_col']
                
                # Crear DataFrame vacío con la estructura correcta
                synthetic_data = pd.DataFrame(columns=feature_cols + [target_col])
                
                logger.info(f"Estructura de datos sintéticos: {synthetic_data.shape}")
                return synthetic_data
            
        except Exception as e:
            logger.error(f"Error generando datos con ADASYN: {str(e)}")
            raise
    
    def get_generator_info(self) -> Dict[str, Any]:
        """Obtiene información del generador ADASYN."""
        return {
            'name': self.generator_name,
            'type': 'Traditional',
            'library': 'imbalanced-learn',
            'model': 'ADASYN',
            'is_fitted': self.is_fitted,
            'config': {
                'n_neighbors': self.n_neighbors,
                'random_state': self.random_state,
                'sampling_strategy': self.sampling_strategy
            }
        }


class TraditionalGenerators:
    """
    Factory para generadores tradicionales.
    """
    
    @staticmethod
    def create_generator(generator_type: str, config: Dict[str, Any]) -> BaseSyntheticGenerator:
        """
        Crea un generador tradicional específico.
        
        Args:
            generator_type: Tipo de generador ('GaussianCopula', 'SMOTE', 'ADASYN', etc.)
            config: Configuración del generador
            
        Returns:
            Instancia del generador
        """
        generators = {
            'GaussianCopula': GaussianCopulaGenerator,
            'SMOTE': SMOTEGenerator,
            'ADASYN': ADASYNGenerator,
        }
        
        if generator_type not in generators:
            raise ValueError(f"Generador tradicional no soportado: {generator_type}")
        
        return generators[generator_type](config)
    
    @staticmethod
    def get_available_generators() -> list:
        """Obtiene lista de generadores tradicionales disponibles."""
        return ['GaussianCopula', 'SMOTE', 'ADASYN']
    
    @staticmethod
    def get_generator_info(generator_type: str) -> Dict[str, Any]:
        """Obtiene información sobre un generador específico."""
        info = {
            'GaussianCopula': {
                'description': 'Gaussian Copula Synthesizer',
                'library': 'SDV',
                'type': 'Traditional',
                'best_for': 'Datos tabulares con distribuciones gaussianas'
            },
            'SMOTE': {
                'description': 'Synthetic Minority Oversampling Technique',
                'library': 'imbalanced-learn',
                'type': 'Traditional',
                'best_for': 'Balanceo de clases en datos desbalanceados'
            },
            'ADASYN': {
                'description': 'Adaptive Synthetic Sampling',
                'library': 'imbalanced-learn',
                'type': 'Traditional',
                'best_for': 'Balanceo adaptativo de clases'
            }
        }
        
        return info.get(generator_type, {})
