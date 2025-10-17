"""
Cargador de datos para el proyecto de grado.
Maneja la carga de datos desde diferentes fuentes (UCI, archivos locales, etc.).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging
from ucimlrepo import fetch_ucirepo

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Clase para cargar datos desde diferentes fuentes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el cargador de datos.
        
        Args:
            config: Configuración del proyecto
        """
        self.config = config
        self.data_paths = config.get('data', {})
        
    def load_uci_dataset(self, dataset_id: int, dataset_name: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Carga un dataset desde UCI Repository.
        
        Args:
            dataset_id: ID del dataset en UCI
            dataset_name: Nombre del dataset (opcional)
            
        Returns:
            Tuple con (features, targets)
        """
        try:
            logger.info(f"Cargando dataset UCI ID: {dataset_id}")
            dataset = fetch_ucirepo(id=dataset_id)
            
            features = dataset.data.features
            targets = dataset.data.targets
            
            logger.info(f"Dataset cargado: {features.shape[0]} filas, {features.shape[1]} features")
            logger.info(f"Target shape: {targets.shape}")
            
            return features, targets
            
        except Exception as e:
            logger.error(f"Error cargando dataset UCI {dataset_id}: {str(e)}")
            raise
    
    def load_credit_datasets(self) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Carga datasets de crédito desde UCI Repository.
        
        Returns:
            Diccionario con datasets de crédito
        """
        credit_datasets = {
            'german_credit': 144,      # German Credit Data
            'australian_credit': 45,   # Australian Credit Approval
            'credit_approval': 27,     # Credit Approval
            'default_credit': 300,     # Default of Credit Card Clients
        }
        
        datasets = {}
        
        for name, dataset_id in credit_datasets.items():
            try:
                logger.info(f"Cargando dataset de crédito: {name}")
                features, targets = self.load_uci_dataset(dataset_id, name)
                datasets[name] = (features, targets)
                
            except Exception as e:
                logger.warning(f"No se pudo cargar {name}: {str(e)}")
                continue
                
        return datasets
    
    def load_local_data(self, file_path: str, file_type: str = 'csv') -> pd.DataFrame:
        """
        Carga datos desde archivo local.
        
        Args:
            file_path: Ruta del archivo
            file_type: Tipo de archivo ('csv', 'excel', 'parquet')
            
        Returns:
            DataFrame con los datos
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
            
            logger.info(f"Cargando archivo local: {file_path}")
            
            if file_type.lower() == 'csv':
                data = pd.read_csv(file_path)
            elif file_type.lower() == 'excel':
                data = pd.read_excel(file_path)
            elif file_type.lower() == 'parquet':
                data = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Tipo de archivo no soportado: {file_type}")
            
            logger.info(f"Datos cargados: {data.shape[0]} filas, {data.shape[1]} columnas")
            return data
            
        except Exception as e:
            logger.error(f"Error cargando archivo local {file_path}: {str(e)}")
            raise
    
    def load_segment_data(self, segment: str) -> pd.DataFrame:
        """
        Carga datos de un segmento específico.
        
        Args:
            segment: Segmento a cargar ('A', 'D', etc.)
            
        Returns:
            DataFrame con los datos del segmento
        """
        try:
            # Construir ruta del archivo
            raw_path = Path(self.data_paths.get('raw_path', 'data/raw'))
            file_path = raw_path / f"segment_{segment}_data.csv"
            
            logger.info(f"Cargando datos del segmento {segment}")
            data = self.load_local_data(file_path, 'csv')
            
            return data
            
        except Exception as e:
            logger.error(f"Error cargando datos del segmento {segment}: {str(e)}")
            raise
    
    def create_sample_data(self, n_samples: int = 5000, n_features: int = 20) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Crea datos de muestra para testing.
        
        Args:
            n_samples: Número de muestras
            n_features: Número de features
            
        Returns:
            Tuple con (features, target)
        """
        logger.info(f"Creando datos de muestra: {n_samples} muestras, {n_features} features")
        
        # Crear features sintéticas
        np.random.seed(42)
        features = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Crear target binario con alguna lógica
        # Simular scoring crediticio
        score = np.random.randn(n_samples)
        target = (score > np.percentile(score, 70)).astype(int)
        
        logger.info(f"Datos de muestra creados: {features.shape}, target: {target.shape}")
        return features, pd.Series(target, name='default')
    
    def save_data(self, data: pd.DataFrame, file_path: str, file_type: str = 'csv') -> None:
        """
        Guarda datos en archivo.
        
        Args:
            data: DataFrame a guardar
            file_path: Ruta del archivo
            file_type: Tipo de archivo
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Guardando datos en: {file_path}")
            
            if file_type.lower() == 'csv':
                data.to_csv(file_path, index=False)
            elif file_type.lower() == 'excel':
                data.to_excel(file_path, index=False)
            elif file_type.lower() == 'parquet':
                data.to_parquet(file_path, index=False)
            else:
                raise ValueError(f"Tipo de archivo no soportado: {file_type}")
                
            logger.info(f"Datos guardados exitosamente: {data.shape}")
            
        except Exception as e:
            logger.error(f"Error guardando datos en {file_path}: {str(e)}")
            raise
