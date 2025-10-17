"""
Generadores GAN para datos sintéticos.
Implementa CTGAN, TVAE, WGAN y otros modelos GAN.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from .base_generator import BaseSyntheticGenerator

logger = logging.getLogger(__name__)


class CTGANGenerator(BaseSyntheticGenerator):
    """
    Generador CTGAN (Conditional Tabular GAN) usando SDV.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "CTGAN")
        
        # Configuración específica de CTGAN
        self.ctgan_config = config.get('synthetic_generators', {}).get('generator_configs', {}).get('CTGAN', {})
        
        # Parámetros por defecto
        self.epochs = self.ctgan_config.get('epochs', 100)
        self.batch_size = self.ctgan_config.get('batch_size', 500)
        self.generator_lr = self.ctgan_config.get('generator_lr', 2e-4)
        self.discriminator_lr = self.ctgan_config.get('discriminator_lr', 2e-4)
        self.generator_decay = self.ctgan_config.get('generator_decay', 1e-6)
        self.discriminator_decay = self.ctgan_config.get('discriminator_decay', 1e-6)
        self.discriminator_steps = self.ctgan_config.get('discriminator_steps', 1)
        self.log_frequency = self.ctgan_config.get('log_frequency', True)
        self.verbose = self.ctgan_config.get('verbose', True)
        self.pac = self.ctgan_config.get('pac', 10)
        self.cuda = self.ctgan_config.get('cuda', False)
        
        logger.info(f"CTGAN configurado: epochs={self.epochs}, batch_size={self.batch_size}")
    
    def fit(self, data: pd.DataFrame, metadata: Optional[Dict] = None) -> None:
        """Entrena el generador CTGAN."""
        try:
            from sdv.single_table import CTGANSynthesizer
            
            if not self.validate_data(data):
                raise ValueError("Datos no válidos para entrenamiento")
            
            logger.info("Iniciando entrenamiento de CTGAN")
            
            # Crear metadatos si no se proporcionan
            if metadata is None:
                metadata = self._create_metadata(data)
            
            # Inicializar CTGAN
            self.generator = CTGANSynthesizer(
                epochs=self.epochs,
                batch_size=self.batch_size,
                generator_lr=self.generator_lr,
                discriminator_lr=self.discriminator_lr,
                generator_decay=self.generator_decay,
                discriminator_decay=self.discriminator_decay,
                discriminator_steps=self.discriminator_steps,
                log_frequency=self.log_frequency,
                verbose=self.verbose,
                pac=self.pac,
                cuda=self.cuda
            )
            
            # Entrenar
            self.generator.fit(data)
            self.metadata = metadata
            self.is_fitted = True
            
            logger.info("CTGAN entrenado exitosamente")
            
        except Exception as e:
            logger.error(f"Error entrenando CTGAN: {str(e)}")
            raise
    
    def generate(self, num_samples: Optional[int] = None) -> pd.DataFrame:
        """Genera datos sintéticos con CTGAN."""
        if not self.is_fitted:
            raise ValueError("El generador debe estar entrenado antes de generar datos")
        
        if num_samples is None:
            num_samples = self.num_samples
        
        try:
            logger.info(f"Generando {num_samples} muestras con CTGAN")
            synthetic_data = self.generator.sample(num_rows=num_samples)
            
            logger.info(f"Datos sintéticos generados: {synthetic_data.shape}")
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Error generando datos con CTGAN: {str(e)}")
            raise
    
    def get_generator_info(self) -> Dict[str, Any]:
        """Obtiene información del generador CTGAN."""
        return {
            'name': self.generator_name,
            'type': 'GAN',
            'library': 'SDV',
            'model': 'CTGAN',
            'is_fitted': self.is_fitted,
            'config': {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'generator_lr': self.generator_lr,
                'discriminator_lr': self.discriminator_lr,
                'pac': self.pac
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


class TVAEGenerator(BaseSyntheticGenerator):
    """
    Generador TVAE (Tabular Variational Autoencoder) usando SDV.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "TVAE")
        
        # Configuración específica de TVAE
        self.tvae_config = config.get('synthetic_generators', {}).get('generator_configs', {}).get('TVAE', {})
        
        # Parámetros por defecto
        self.epochs = self.tvae_config.get('epochs', 100)
        self.batch_size = self.tvae_config.get('batch_size', 500)
        self.lr = self.tvae_config.get('lr', 2e-3)
        self.loss_factor = self.tvae_config.get('loss_factor', 2)
        self.cuda = self.tvae_config.get('cuda', False)
        
        logger.info(f"TVAE configurado: epochs={self.epochs}, batch_size={self.batch_size}")
    
    def fit(self, data: pd.DataFrame, metadata: Optional[Dict] = None) -> None:
        """Entrena el generador TVAE."""
        try:
            from sdv.single_table import TVAESynthesizer
            
            if not self.validate_data(data):
                raise ValueError("Datos no válidos para entrenamiento")
            
            logger.info("Iniciando entrenamiento de TVAE")
            
            # Crear metadatos si no se proporcionan
            if metadata is None:
                metadata = self._create_metadata(data)
            
            # Inicializar TVAE
            self.generator = TVAESynthesizer(
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                loss_factor=self.loss_factor,
                cuda=self.cuda
            )
            
            # Entrenar
            self.generator.fit(data)
            self.metadata = metadata
            self.is_fitted = True
            
            logger.info("TVAE entrenado exitosamente")
            
        except Exception as e:
            logger.error(f"Error entrenando TVAE: {str(e)}")
            raise
    
    def generate(self, num_samples: Optional[int] = None) -> pd.DataFrame:
        """Genera datos sintéticos con TVAE."""
        if not self.is_fitted:
            raise ValueError("El generador debe estar entrenado antes de generar datos")
        
        if num_samples is None:
            num_samples = self.num_samples
        
        try:
            logger.info(f"Generando {num_samples} muestras con TVAE")
            synthetic_data = self.generator.sample(num_rows=num_samples)
            
            logger.info(f"Datos sintéticos generados: {synthetic_data.shape}")
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Error generando datos con TVAE: {str(e)}")
            raise
    
    def get_generator_info(self) -> Dict[str, Any]:
        """Obtiene información del generador TVAE."""
        return {
            'name': self.generator_name,
            'type': 'GAN',
            'library': 'SDV',
            'model': 'TVAE',
            'is_fitted': self.is_fitted,
            'config': {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'loss_factor': self.loss_factor
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


class GANGenerators:
    """
    Factory para generadores GAN.
    """
    
    @staticmethod
    def create_generator(generator_type: str, config: Dict[str, Any]) -> BaseSyntheticGenerator:
        """
        Crea un generador GAN específico.
        
        Args:
            generator_type: Tipo de generador ('CTGAN', 'TVAE', etc.)
            config: Configuración del generador
            
        Returns:
            Instancia del generador
        """
        generators = {
            'CTGAN': CTGANGenerator,
            'TVAE': TVAEGenerator,
        }
        
        if generator_type not in generators:
            raise ValueError(f"Generador GAN no soportado: {generator_type}")
        
        return generators[generator_type](config)
    
    @staticmethod
    def get_available_generators() -> list:
        """Obtiene lista de generadores GAN disponibles."""
        return ['CTGAN', 'TVAE']
    
    @staticmethod
    def get_generator_info(generator_type: str) -> Dict[str, Any]:
        """Obtiene información sobre un generador específico."""
        info = {
            'CTGAN': {
                'description': 'Conditional Tabular GAN',
                'library': 'SDV',
                'type': 'GAN',
                'best_for': 'Datos tabulares con variables categóricas y numéricas'
            },
            'TVAE': {
                'description': 'Tabular Variational Autoencoder',
                'library': 'SDV', 
                'type': 'GAN',
                'best_for': 'Datos tabulares con alta dimensionalidad'
            }
        }
        
        return info.get(generator_type, {})
