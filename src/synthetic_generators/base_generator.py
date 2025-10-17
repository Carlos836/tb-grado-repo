"""
Generador base para datos sintéticos.
Define la interfaz común para todos los generadores.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseSyntheticGenerator(ABC):
    """
    Clase base abstracta para todos los generadores de datos sintéticos.
    """
    
    def __init__(self, config: Dict[str, Any], generator_name: str):
        """
        Inicializa el generador base.
        
        Args:
            config: Configuración del generador
            generator_name: Nombre del generador
        """
        self.config = config
        self.generator_name = generator_name
        self.is_fitted = False
        self.metadata = None
        self.generator = None
        
        # Configuración específica del generador
        self.generator_config = config.get('synthetic_generators', {}).get('generation', {})
        self.random_state = self.generator_config.get('random_state', 42)
        self.num_samples = self.generator_config.get('num_samples', 10000)
        
        logger.info(f"Generador {generator_name} inicializado")
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, metadata: Optional[Dict] = None) -> None:
        """
        Entrena el generador con los datos reales.
        
        Args:
            data: Datos reales para entrenar
            metadata: Metadatos del dataset (opcional)
        """
        pass
    
    @abstractmethod
    def generate(self, num_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Genera datos sintéticos.
        
        Args:
            num_samples: Número de muestras a generar
            
        Returns:
            DataFrame con datos sintéticos
        """
        pass
    
    @abstractmethod
    def get_generator_info(self) -> Dict[str, Any]:
        """
        Obtiene información del generador.
        
        Returns:
            Diccionario con información del generador
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Valida que los datos sean adecuados para el generador.
        
        Args:
            data: Datos a validar
            
        Returns:
            True si los datos son válidos
        """
        if data is None or data.empty:
            logger.error("Datos vacíos o nulos")
            return False
        
        if len(data.columns) == 0:
            logger.error("Datos sin columnas")
            return False
        
        if len(data) < 10:
            logger.warning(f"Datos con muy pocas muestras: {len(data)}")
        
        logger.info(f"Datos validados: {data.shape[0]} filas, {data.shape[1]} columnas")
        return True
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara los datos para el generador.
        
        Args:
            data: Datos a preparar
            
        Returns:
            Datos preparados
        """
        # Copiar datos para no modificar el original
        prepared_data = data.copy()
        
        # Manejar valores faltantes
        if prepared_data.isnull().any().any():
            logger.info("Manejando valores faltantes")
            prepared_data = prepared_data.fillna(prepared_data.median())
        
        # Convertir tipos de datos
        for col in prepared_data.columns:
            if prepared_data[col].dtype == 'object':
                # Convertir categóricas a numéricas
                prepared_data[col] = pd.Categorical(prepared_data[col]).codes
        
        logger.info(f"Datos preparados: {prepared_data.shape}")
        return prepared_data
    
    def save_generator(self, file_path: str) -> None:
        """
        Guarda el generador entrenado.
        
        Args:
            file_path: Ruta donde guardar
        """
        import joblib
        
        if not self.is_fitted:
            raise ValueError("El generador debe estar entrenado antes de guardarlo")
        
        generator_data = {
            'generator': self.generator,
            'metadata': self.metadata,
            'config': self.config,
            'generator_name': self.generator_name,
            'is_fitted': self.is_fitted
        }
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(generator_data, file_path)
        logger.info(f"Generador guardado en: {file_path}")
    
    def load_generator(self, file_path: str) -> None:
        """
        Carga un generador entrenado.
        
        Args:
            file_path: Ruta del archivo
        """
        import joblib
        
        generator_data = joblib.load(file_path)
        
        self.generator = generator_data['generator']
        self.metadata = generator_data['metadata']
        self.config = generator_data['config']
        self.generator_name = generator_data['generator_name']
        self.is_fitted = generator_data['is_fitted']
        
        logger.info(f"Generador cargado desde: {file_path}")
    
    def get_metadata(self) -> Optional[Dict]:
        """
        Obtiene metadatos del generador.
        
        Returns:
            Metadatos del generador
        """
        return self.metadata
    
    def set_metadata(self, metadata: Dict) -> None:
        """
        Establece metadatos del generador.
        
        Args:
            metadata: Metadatos a establecer
        """
        self.metadata = metadata
        logger.info("Metadatos establecidos")
    
    def __str__(self) -> str:
        """Representación string del generador."""
        return f"{self.generator_name}(fitted={self.is_fitted})"
    
    def __repr__(self) -> str:
        """Representación detallada del generador."""
        return f"{self.__class__.__name__}(name='{self.generator_name}', fitted={self.is_fitted})"
