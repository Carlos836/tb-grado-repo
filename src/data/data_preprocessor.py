"""
Preprocesador de datos para el proyecto de grado.
Maneja la limpieza, transformación y preparación de datos.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Clase para preprocesar datos de scoring crediticio.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el preprocesador.
        
        Args:
            config: Configuración del proyecto
        """
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = None
        
    def detect_data_types(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Detecta tipos de datos en el DataFrame.
        
        Args:
            data: DataFrame a analizar
            
        Returns:
            Diccionario con tipos de datos por categoría
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Detectar columnas binarias
        binary_cols = []
        for col in numeric_cols:
            if data[col].nunique() == 2:
                binary_cols.append(col)
        
        # Remover columnas binarias de numéricas
        numeric_cols = [col for col in numeric_cols if col not in binary_cols]
        
        data_types = {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'binary': binary_cols,
            'datetime': datetime_cols
        }
        
        logger.info(f"Tipos de datos detectados:")
        for dtype, cols in data_types.items():
            logger.info(f"  {dtype}: {len(cols)} columnas")
            
        return data_types
    
    def handle_missing_values(self, data: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """
        Maneja valores faltantes en el DataFrame.
        
        Args:
            data: DataFrame con valores faltantes
            strategy: Estrategia de imputación ('mean', 'median', 'mode', 'drop')
            
        Returns:
            DataFrame sin valores faltantes
        """
        logger.info(f"Manejando valores faltantes con estrategia: {strategy}")
        
        data_clean = data.copy()
        
        # Detectar columnas con valores faltantes
        missing_cols = data_clean.columns[data_clean.isnull().any()].tolist()
        
        if not missing_cols:
            logger.info("No se encontraron valores faltantes")
            return data_clean
        
        logger.info(f"Columnas con valores faltantes: {missing_cols}")
        
        for col in missing_cols:
            missing_count = data_clean[col].isnull().sum()
            missing_pct = (missing_count / len(data_clean)) * 100
            
            logger.info(f"  {col}: {missing_count} ({missing_pct:.2f}%)")
            
            if strategy == 'drop':
                # Eliminar filas con valores faltantes
                data_clean = data_clean.dropna(subset=[col])
            else:
                # Imputar valores
                if data_clean[col].dtype in ['object', 'category']:
                    # Para variables categóricas, usar moda
                    imputer = SimpleImputer(strategy='most_frequent')
                else:
                    # Para variables numéricas, usar la estrategia especificada
                    imputer = SimpleImputer(strategy=strategy)
                
                data_clean[col] = imputer.fit_transform(data_clean[[col]]).flatten()
        
        logger.info(f"Datos después de manejar valores faltantes: {data_clean.shape}")
        return data_clean
    
    def encode_categorical_variables(self, data: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Codifica variables categóricas.
        
        Args:
            data: DataFrame con variables categóricas
            target_col: Nombre de la columna target (opcional)
            
        Returns:
            DataFrame con variables codificadas
        """
        logger.info("Codificando variables categóricas")
        
        data_encoded = data.copy()
        data_types = self.detect_data_types(data_encoded)
        
        # Codificar variables categóricas
        for col in data_types['categorical']:
            if col == target_col:
                continue
                
            # Usar LabelEncoder para variables categóricas
            le = LabelEncoder()
            data_encoded[col] = le.fit_transform(data_encoded[col].astype(str))
            self.encoders[col] = le
            
            logger.info(f"  {col}: {len(le.classes_)} categorías codificadas")
        
        return data_encoded
    
    def scale_numeric_variables(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Escala variables numéricas.
        
        Args:
            data: DataFrame con variables numéricas
            fit: Si debe ajustar el scaler (True) o usar uno existente (False)
            
        Returns:
            DataFrame con variables escaladas
        """
        logger.info("Escalando variables numéricas")
        
        data_scaled = data.copy()
        data_types = self.detect_data_types(data_scaled)
        
        numeric_cols = data_types['numeric']
        
        if not numeric_cols:
            logger.info("No hay variables numéricas para escalar")
            return data_scaled
        
        if fit:
            # Ajustar scaler
            scaler = StandardScaler()
            data_scaled[numeric_cols] = scaler.fit_transform(data_scaled[numeric_cols])
            self.scalers['numeric'] = scaler
            logger.info(f"Scaler ajustado para {len(numeric_cols)} variables numéricas")
        else:
            # Usar scaler existente
            if 'numeric' in self.scalers:
                scaler = self.scalers['numeric']
                data_scaled[numeric_cols] = scaler.transform(data_scaled[numeric_cols])
                logger.info(f"Scaler aplicado a {len(numeric_cols)} variables numéricas")
            else:
                logger.warning("No hay scaler disponible para variables numéricas")
        
        return data_scaled
    
    def create_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features de ingeniería.
        
        Args:
            data: DataFrame base
            
        Returns:
            DataFrame con features adicionales
        """
        logger.info("Creando features de ingeniería")
        
        data_eng = data.copy()
        
        # Crear features de interacción (ejemplo)
        numeric_cols = data_eng.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Crear feature de interacción entre las dos primeras variables numéricas
            col1, col2 = numeric_cols[0], numeric_cols[1]
            data_eng[f'{col1}_x_{col2}'] = data_eng[col1] * data_eng[col2]
            logger.info(f"Feature de interacción creada: {col1}_x_{col2}")
        
        # Crear features polinómicas (ejemplo)
        if len(numeric_cols) >= 1:
            col = numeric_cols[0]
            data_eng[f'{col}_squared'] = data_eng[col] ** 2
            logger.info(f"Feature polinómica creada: {col}_squared")
        
        logger.info(f"Features de ingeniería creadas. Nuevo shape: {data_eng.shape}")
        return data_eng
    
    def preprocess_data(self, data: pd.DataFrame, target_col: str = None, 
                       fit: bool = True) -> pd.DataFrame:
        """
        Preprocesa datos completos.
        
        Args:
            data: DataFrame a preprocesar
            target_col: Nombre de la columna target
            fit: Si debe ajustar los transformadores
            
        Returns:
            DataFrame preprocesado
        """
        logger.info("Iniciando preprocesamiento completo de datos")
        
        # 1. Manejar valores faltantes
        data_clean = self.handle_missing_values(data)
        
        # 2. Codificar variables categóricas
        data_encoded = self.encode_categorical_variables(data_clean, target_col)
        
        # 3. Escalar variables numéricas
        data_scaled = self.scale_numeric_variables(data_encoded, fit=fit)
        
        # 4. Crear features de ingeniería
        data_eng = self.create_feature_engineering(data_scaled)
        
        # Guardar nombres de features
        if fit:
            self.feature_names = data_eng.columns.tolist()
            if target_col and target_col in self.feature_names:
                self.feature_names.remove(target_col)
        
        logger.info(f"Preprocesamiento completado. Shape final: {data_eng.shape}")
        return data_eng
    
    def get_feature_names(self) -> List[str]:
        """
        Obtiene nombres de features.
        
        Returns:
            Lista de nombres de features
        """
        return self.feature_names if self.feature_names else []
    
    def save_preprocessor(self, file_path: str) -> None:
        """
        Guarda el preprocesador.
        
        Args:
            file_path: Ruta donde guardar
        """
        import joblib
        
        preprocessor_data = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'imputers': self.imputers,
            'feature_names': self.feature_names
        }
        
        joblib.dump(preprocessor_data, file_path)
        logger.info(f"Preprocesador guardado en: {file_path}")
    
    def load_preprocessor(self, file_path: str) -> None:
        """
        Carga el preprocesador.
        
        Args:
            file_path: Ruta del archivo
        """
        import joblib
        
        preprocessor_data = joblib.load(file_path)
        
        self.scalers = preprocessor_data['scalers']
        self.encoders = preprocessor_data['encoders']
        self.imputers = preprocessor_data['imputers']
        self.feature_names = preprocessor_data['feature_names']
        
        logger.info(f"Preprocesador cargado desde: {file_path}")
