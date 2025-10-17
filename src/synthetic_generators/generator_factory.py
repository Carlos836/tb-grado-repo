"""
Factory para crear generadores de datos sintéticos.
Maneja la creación de generadores GAN y tradicionales.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
from .base_generator import BaseSyntheticGenerator
from .gan_generators import GANGenerators
from .traditional_generators import TraditionalGenerators

logger = logging.getLogger(__name__)


class GeneratorFactory:
    """
    Factory para crear y gestionar generadores de datos sintéticos.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el factory de generadores.
        
        Args:
            config: Configuración del proyecto
        """
        self.config = config
        self.generators = {}
        self.generator_results = {}
        
        # Configuración de generadores
        self.synthetic_config = config.get('synthetic_generators', {})
        self.gan_models = self.synthetic_config.get('gan_models', [])
        self.traditional_models = self.synthetic_config.get('traditional_models', [])
        
        logger.info(f"GeneratorFactory inicializado")
        logger.info(f"  GAN models: {self.gan_models}")
        logger.info(f"  Traditional models: {self.traditional_models}")
    
    def create_generator(self, generator_type: str, 
                        generator_name: str = None) -> BaseSyntheticGenerator:
        """
        Crea un generador específico.
        
        Args:
            generator_type: Tipo de generador ('CTGAN', 'TVAE', 'GaussianCopula', etc.)
            generator_name: Nombre personalizado del generador (opcional)
            
        Returns:
            Instancia del generador
        """
        if generator_name is None:
            generator_name = generator_type
        
        try:
            # Intentar crear generador GAN
            if generator_type in self.gan_models:
                generator = GANGenerators.create_generator(generator_type, self.config)
                logger.info(f"Generador GAN creado: {generator_type}")
                return generator
            
            # Intentar crear generador tradicional
            elif generator_type in self.traditional_models:
                generator = TraditionalGenerators.create_generator(generator_type, self.config)
                logger.info(f"Generador tradicional creado: {generator_type}")
                return generator
            
            else:
                raise ValueError(f"Tipo de generador no soportado: {generator_type}")
                
        except Exception as e:
            logger.error(f"Error creando generador {generator_type}: {str(e)}")
            raise
    
    def create_all_generators(self) -> Dict[str, BaseSyntheticGenerator]:
        """
        Crea todos los generadores configurados.
        
        Returns:
            Diccionario con todos los generadores
        """
        all_generators = {}
        
        # Crear generadores GAN
        for gan_model in self.gan_models:
            try:
                generator = self.create_generator(gan_model)
                all_generators[gan_model] = generator
                logger.info(f"Generador GAN {gan_model} creado exitosamente")
            except Exception as e:
                logger.warning(f"No se pudo crear generador GAN {gan_model}: {e}")
        
        # Crear generadores tradicionales
        for traditional_model in self.traditional_models:
            try:
                generator = self.create_generator(traditional_model)
                all_generators[traditional_model] = generator
                logger.info(f"Generador tradicional {traditional_model} creado exitosamente")
            except Exception as e:
                logger.warning(f"No se pudo crear generador tradicional {traditional_model}: {e}")
        
        self.generators = all_generators
        logger.info(f"Total de generadores creados: {len(all_generators)}")
        
        return all_generators
    
    def train_generator(self, generator: BaseSyntheticGenerator, 
                       data: pd.DataFrame, 
                       metadata: Optional[Dict] = None) -> None:
        """
        Entrena un generador específico.
        
        Args:
            generator: Generador a entrenar
            data: Datos de entrenamiento
            metadata: Metadatos del dataset
        """
        try:
            generator_name = generator.generator_name
            logger.info(f"Entrenando generador: {generator_name}")
            
            # Entrenar generador
            generator.fit(data, metadata)
            
            # Guardar en diccionario de generadores
            self.generators[generator_name] = generator
            
            logger.info(f"Generador {generator_name} entrenado exitosamente")
            
        except Exception as e:
            logger.error(f"Error entrenando generador {generator_name}: {str(e)}")
            raise
    
    def train_all_generators(self, data: pd.DataFrame, 
                           metadata: Optional[Dict] = None) -> Dict[str, BaseSyntheticGenerator]:
        """
        Entrena todos los generadores con los datos proporcionados.
        
        Args:
            data: Datos de entrenamiento
            metadata: Metadatos del dataset
            
        Returns:
            Diccionario con generadores entrenados
        """
        logger.info("Iniciando entrenamiento de todos los generadores")
        
        trained_generators = {}
        
        for generator_name, generator in self.generators.items():
            try:
                logger.info(f"Entrenando {generator_name}...")
                generator.fit(data, metadata)
                trained_generators[generator_name] = generator
                logger.info(f"✅ {generator_name} entrenado exitosamente")
                
            except Exception as e:
                logger.error(f"❌ Error entrenando {generator_name}: {str(e)}")
                continue
        
        logger.info(f"Entrenamiento completado: {len(trained_generators)}/{len(self.generators)} generadores exitosos")
        
        return trained_generators
    
    def generate_synthetic_data(self, generator_name: str, 
                               num_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Genera datos sintéticos con un generador específico.
        
        Args:
            generator_name: Nombre del generador
            num_samples: Número de muestras a generar
            
        Returns:
            DataFrame con datos sintéticos
        """
        if generator_name not in self.generators:
            raise ValueError(f"Generador {generator_name} no encontrado")
        
        generator = self.generators[generator_name]
        
        if not generator.is_fitted:
            raise ValueError(f"Generador {generator_name} no está entrenado")
        
        try:
            logger.info(f"Generando datos sintéticos con {generator_name}")
            synthetic_data = generator.generate(num_samples)
            
            logger.info(f"Datos sintéticos generados: {synthetic_data.shape}")
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Error generando datos con {generator_name}: {str(e)}")
            raise
    
    def generate_all_synthetic_data(self, num_samples: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Genera datos sintéticos con todos los generadores entrenados.
        
        Args:
            num_samples: Número de muestras a generar por generador
            
        Returns:
            Diccionario con datos sintéticos por generador
        """
        logger.info("Generando datos sintéticos con todos los generadores")
        
        all_synthetic_data = {}
        
        for generator_name, generator in self.generators.items():
            if generator.is_fitted:
                try:
                    logger.info(f"Generando con {generator_name}...")
                    synthetic_data = generator.generate(num_samples)
                    all_synthetic_data[generator_name] = synthetic_data
                    logger.info(f"✅ {generator_name}: {synthetic_data.shape}")
                    
                except Exception as e:
                    logger.error(f"❌ Error generando con {generator_name}: {str(e)}")
                    continue
            else:
                logger.warning(f"Generador {generator_name} no está entrenado")
        
        logger.info(f"Generación completada: {len(all_synthetic_data)} datasets sintéticos")
        
        return all_synthetic_data
    
    def get_generator_info(self, generator_name: str = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Obtiene información de un generador específico o de todos.
        
        Args:
            generator_name: Nombre del generador (opcional)
            
        Returns:
            Información del generador o de todos los generadores
        """
        if generator_name:
            if generator_name not in self.generators:
                raise ValueError(f"Generador {generator_name} no encontrado")
            
            return self.generators[generator_name].get_generator_info()
        
        else:
            all_info = {}
            for name, generator in self.generators.items():
                all_info[name] = generator.get_generator_info()
            
            return all_info
    
    def get_available_generators(self) -> Dict[str, List[str]]:
        """
        Obtiene lista de generadores disponibles.
        
        Returns:
            Diccionario con generadores disponibles por tipo
        """
        return {
            'gan_generators': GANGenerators.get_available_generators(),
            'traditional_generators': TraditionalGenerators.get_available_generators(),
            'configured_gan': self.gan_models,
            'configured_traditional': self.traditional_models
        }
    
    def save_generators(self, base_path: str) -> None:
        """
        Guarda todos los generadores entrenados.
        
        Args:
            base_path: Ruta base donde guardar
        """
        from pathlib import Path
        
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Guardando generadores en: {base_path}")
        
        for generator_name, generator in self.generators.items():
            if generator.is_fitted:
                file_path = base_path / f"{generator_name}_generator.pkl"
                generator.save_generator(str(file_path))
                logger.info(f"  {generator_name} guardado en {file_path}")
    
    def load_generators(self, base_path: str) -> None:
        """
        Carga generadores desde archivos.
        
        Args:
            base_path: Ruta base donde están los archivos
        """
        from pathlib import Path
        
        base_path = Path(base_path)
        
        if not base_path.exists():
            logger.warning(f"Directorio {base_path} no existe")
            return
        
        logger.info(f"Cargando generadores desde: {base_path}")
        
        # Buscar archivos de generadores
        generator_files = list(base_path.glob("*_generator.pkl"))
        
        for file_path in generator_files:
            try:
                generator_name = file_path.stem.replace("_generator", "")
                
                # Crear generador temporal para cargar
                temp_generator = self.create_generator(generator_name)
                temp_generator.load_generator(str(file_path))
                
                self.generators[generator_name] = temp_generator
                logger.info(f"  {generator_name} cargado desde {file_path}")
                
            except Exception as e:
                logger.warning(f"No se pudo cargar generador desde {file_path}: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del entrenamiento.
        
        Returns:
            Resumen del entrenamiento
        """
        summary = {
            'total_generators': len(self.generators),
            'trained_generators': sum(1 for g in self.generators.values() if g.is_fitted),
            'generator_types': {},
            'generator_status': {}
        }
        
        for name, generator in self.generators.items():
            info = generator.get_generator_info()
            generator_type = info.get('type', 'Unknown')
            
            if generator_type not in summary['generator_types']:
                summary['generator_types'][generator_type] = 0
            summary['generator_types'][generator_type] += 1
            
            summary['generator_status'][name] = {
                'type': generator_type,
                'library': info.get('library', 'Unknown'),
                'model': info.get('model', 'Unknown'),
                'is_fitted': generator.is_fitted
            }
        
        return summary
    
    def __str__(self) -> str:
        """Representación string del factory."""
        summary = self.get_training_summary()
        return f"GeneratorFactory({summary['trained_generators']}/{summary['total_generators']} trained)"
    
    def __repr__(self) -> str:
        """Representación detallada del factory."""
        return f"GeneratorFactory(generators={len(self.generators)}, trained={sum(1 for g in self.generators.values() if g.is_fitted)})"
