"""
Divisor de datos para el proyecto de grado.
Maneja la división de datos en train/validation/test.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Clase para dividir datos en conjuntos de entrenamiento, validación y prueba.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el divisor de datos.
        
        Args:
            config: Configuración del proyecto
        """
        self.config = config
        self.data_config = config.get('data', {})
        
        # Configuración de división
        self.train_ratio = self.data_config.get('train_ratio', 0.6)
        self.validation_ratio = self.data_config.get('validation_ratio', 0.2)
        self.test_ratio = self.data_config.get('test_ratio', 0.2)
        
        # Configuración de validación para sintetizadores
        self.synthetic_validation_ratio = self.data_config.get('synthetic_validation_ratio', 0.15)
        
        # Configuración de validación cruzada
        self.cv_folds = config.get('ml_models', {}).get('training', {}).get('cv_folds', 5)
        self.random_state = config.get('ml_models', {}).get('training', {}).get('random_state', 42)
        
        logger.info(f"DataSplitter inicializado:")
        logger.info(f"  Train: {self.train_ratio}, Validation: {self.validation_ratio}, Test: {self.test_ratio}")
        logger.info(f"  Synthetic validation: {self.synthetic_validation_ratio}")
        logger.info(f"  CV folds: {self.cv_folds}")
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   stratify: bool = True) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Divide datos en train/validation/test.
        
        Args:
            X: Features
            y: Target
            stratify: Si debe estratificar por target
            
        Returns:
            Diccionario con splits
        """
        logger.info("Dividiendo datos en train/validation/test")
        
        # Verificar que las proporciones sumen 1
        total_ratio = self.train_ratio + self.validation_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Las proporciones deben sumar 1.0, actual: {total_ratio}")
        
        # Primera división: train vs (validation + test)
        train_size = self.train_ratio
        val_test_size = self.validation_ratio + self.test_ratio
        
        X_train, X_val_test, y_train, y_val_test = train_test_split(
            X, y,
            test_size=val_test_size,
            random_state=self.random_state,
            stratify=y if stratify else None
        )
        
        # Segunda división: validation vs test
        val_size = self.validation_ratio / val_test_size
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test, y_val_test,
            test_size=(1 - val_size),
            random_state=self.random_state,
            stratify=y_val_test if stratify else None
        )
        
        splits = {
            'train': (X_train, y_train),
            'validation': (X_val, y_val),
            'test': (X_test, y_test)
        }
        
        # Log de información
        for split_name, (X_split, y_split) in splits.items():
            logger.info(f"  {split_name}: {X_split.shape[0]} muestras")
            if stratify:
                unique, counts = np.unique(y_split, return_counts=True)
                for val, count in zip(unique, counts):
                    pct = (count / len(y_split)) * 100
                    logger.info(f"    Clase {val}: {count} ({pct:.1f}%)")
        
        return splits
    
    def split_for_synthetic_validation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Divide datos para validación de sintetizadores.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Diccionario con splits para sintetizadores
        """
        logger.info("Dividiendo datos para validación de sintetizadores")
        
        # Dividir en datos para entrenar sintetizadores y datos para validación
        X_synth_train, X_synth_val, y_synth_train, y_synth_val = train_test_split(
            X, y,
            test_size=self.synthetic_validation_ratio,
            random_state=self.random_state,
            stratify=y
        )
        
        splits = {
            'synthetic_train': (X_synth_train, y_synth_train),
            'synthetic_validation': (X_synth_val, y_synth_val)
        }
        
        logger.info(f"  Synthetic train: {X_synth_train.shape[0]} muestras")
        logger.info(f"  Synthetic validation: {X_synth_val.shape[0]} muestras")
        
        return splits
    
    def create_cv_splits(self, X: pd.DataFrame, y: pd.Series) -> list:
        """
        Crea splits para validación cruzada.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Lista de splits para CV
        """
        logger.info(f"Creando {self.cv_folds} splits para validación cruzada")
        
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_splits = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            cv_splits.append({
                'fold': fold,
                'train': (X_train_fold, y_train_fold),
                'validation': (X_val_fold, y_val_fold)
            })
            
            logger.info(f"  Fold {fold}: train={len(train_idx)}, val={len(val_idx)}")
        
        return cv_splits
    
    def create_time_series_splits(self, X: pd.DataFrame, y: pd.Series, 
                                 time_col: str = None, n_splits: int = 5) -> list:
        """
        Crea splits para series temporales (útil para datos de crédito).
        
        Args:
            X: Features
            y: Target
            time_col: Columna de tiempo (opcional)
            n_splits: Número de splits
            
        Returns:
            Lista de splits temporales
        """
        logger.info(f"Creando {n_splits} splits temporales")
        
        if time_col and time_col in X.columns:
            # Ordenar por tiempo
            X_sorted = X.sort_values(time_col)
            y_sorted = y.loc[X_sorted.index]
        else:
            # Usar índice como proxy de tiempo
            X_sorted = X.sort_index()
            y_sorted = y.loc[X_sorted.index]
        
        # Crear splits temporales
        n_samples = len(X_sorted)
        split_size = n_samples // n_splits
        
        time_splits = []
        
        for i in range(n_splits):
            # Definir rangos
            train_end = (i + 1) * split_size
            val_start = train_end
            val_end = min((i + 2) * split_size, n_samples)
            
            if val_start >= n_samples:
                break
            
            # Crear splits
            X_train = X_sorted.iloc[:train_end]
            X_val = X_sorted.iloc[val_start:val_end]
            y_train = y_sorted.iloc[:train_end]
            y_val = y_sorted.iloc[val_start:val_end]
            
            time_splits.append({
                'fold': i,
                'train': (X_train, y_train),
                'validation': (X_val, y_val)
            })
            
            logger.info(f"  Time split {i}: train={len(X_train)}, val={len(X_val)}")
        
        return time_splits
    
    def get_data_info(self, splits: Dict[str, Tuple[pd.DataFrame, pd.Series]]) -> Dict[str, Any]:
        """
        Obtiene información sobre los splits.
        
        Args:
            splits: Diccionario con splits
            
        Returns:
            Información sobre los splits
        """
        info = {}
        
        for split_name, (X, y) in splits.items():
            info[split_name] = {
                'n_samples': len(X),
                'n_features': X.shape[1],
                'target_distribution': y.value_counts().to_dict(),
                'target_balance': y.value_counts(normalize=True).to_dict()
            }
        
        return info
    
    def save_splits(self, splits: Dict[str, Tuple[pd.DataFrame, pd.Series]], 
                   base_path: str) -> None:
        """
        Guarda splits en archivos.
        
        Args:
            splits: Diccionario con splits
            base_path: Ruta base para guardar
        """
        from pathlib import Path
        
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Guardando splits en: {base_path}")
        
        for split_name, (X, y) in splits.items():
            # Guardar features
            X_path = base_path / f"{split_name}_X.csv"
            X.to_csv(X_path, index=False)
            
            # Guardar target
            y_path = base_path / f"{split_name}_y.csv"
            y.to_csv(y_path, index=False)
            
            logger.info(f"  {split_name}: X={X_path}, y={y_path}")
    
    def load_splits(self, base_path: str) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Carga splits desde archivos.
        
        Args:
            base_path: Ruta base donde están los archivos
            
        Returns:
            Diccionario con splits
        """
        from pathlib import Path
        
        base_path = Path(base_path)
        splits = {}
        
        logger.info(f"Cargando splits desde: {base_path}")
        
        # Buscar archivos de splits
        for file_path in base_path.glob("*_X.csv"):
            split_name = file_path.stem.replace("_X", "")
            y_path = base_path / f"{split_name}_y.csv"
            
            if y_path.exists():
                X = pd.read_csv(file_path)
                y = pd.read_csv(y_path, squeeze=True)
                splits[split_name] = (X, y)
                
                logger.info(f"  {split_name}: {X.shape[0]} muestras cargadas")
        
        return splits
