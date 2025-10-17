"""
Factory para crear y gestionar modelos de scoring crediticio.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
from .base_model import BaseCreditScoringModel
from .credit_scoring_models import CreditScoringModels

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory para crear y gestionar modelos de scoring crediticio.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el factory de modelos.
        
        Args:
            config: Configuración del proyecto
        """
        self.config = config
        self.models = {}
        self.model_results = {}
        
        # Configuración de modelos
        self.ml_config = config.get('ml_models', {})
        self.active_models = self.ml_config.get('models', [])
        
        logger.info(f"ModelFactory inicializado")
        logger.info(f"  Active models: {self.active_models}")
    
    def create_model(self, model_type: str, 
                    model_name: str = None) -> BaseCreditScoringModel:
        """
        Crea un modelo específico.
        
        Args:
            model_type: Tipo de modelo ('XGBoost', 'CatBoost', etc.)
            model_name: Nombre personalizado del modelo (opcional)
            
        Returns:
            Instancia del modelo
        """
        if model_name is None:
            model_name = model_type
        
        try:
            model = CreditScoringModels.create_model(model_type, self.config)
            logger.info(f"Modelo creado: {model_type}")
            return model
            
        except Exception as e:
            logger.error(f"Error creando modelo {model_type}: {str(e)}")
            raise
    
    def create_all_models(self) -> Dict[str, BaseCreditScoringModel]:
        """
        Crea todos los modelos configurados.
        
        Returns:
            Diccionario con todos los modelos
        """
        all_models = {}
        
        for model_type in self.active_models:
            try:
                model = self.create_model(model_type)
                all_models[model_type] = model
                logger.info(f"Modelo {model_type} creado exitosamente")
            except Exception as e:
                logger.warning(f"No se pudo crear modelo {model_type}: {e}")
        
        self.models = all_models
        logger.info(f"Total de modelos creados: {len(all_models)}")
        
        return all_models
    
    def train_model(self, model: BaseCreditScoringModel, 
                   X_train: pd.DataFrame, y_train: pd.Series,
                   **fit_params) -> None:
        """
        Entrena un modelo específico.
        
        Args:
            model: Modelo a entrenar
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            **fit_params: Parámetros adicionales de entrenamiento
        """
        try:
            model_name = model.model_name
            logger.info(f"Entrenando modelo: {model_name}")
            
            # Entrenar modelo
            model.fit(X_train, y_train, **fit_params)
            
            # Guardar en diccionario de modelos
            self.models[model_name] = model
            
            logger.info(f"Modelo {model_name} entrenado exitosamente")
            
        except Exception as e:
            logger.error(f"Error entrenando modelo {model_name}: {str(e)}")
            raise
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        **fit_params) -> Dict[str, BaseCreditScoringModel]:
        """
        Entrena todos los modelos con los datos proporcionados.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            **fit_params: Parámetros adicionales de entrenamiento
            
        Returns:
            Diccionario con modelos entrenados
        """
        logger.info("Iniciando entrenamiento de todos los modelos")
        
        trained_models = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Entrenando {model_name}...")
                model.fit(X_train, y_train, **fit_params)
                trained_models[model_name] = model
                logger.info(f"✅ {model_name} entrenado exitosamente")
                
            except Exception as e:
                logger.error(f"❌ Error entrenando {model_name}: {str(e)}")
                continue
        
        logger.info(f"Entrenamiento completado: {len(trained_models)}/{len(self.models)} modelos exitosos")
        
        return trained_models
    
    def evaluate_model(self, model: BaseCreditScoringModel,
                      X_test: pd.DataFrame, y_test: pd.Series,
                      X_train: Optional[pd.DataFrame] = None,
                      y_train: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Evalúa un modelo específico.
        
        Args:
            model: Modelo a evaluar
            X_test: Features de prueba
            y_test: Target de prueba
            X_train: Features de entrenamiento (opcional, para PSI)
            y_train: Target de entrenamiento (opcional, para PSI)
            
        Returns:
            Resultados de evaluación
        """
        try:
            from .model_evaluator import CreditModelEvaluator
            
            evaluator = CreditModelEvaluator(self.config)
            results = evaluator.evaluate_model(model, X_test, y_test, X_train, y_train)
            
            # Guardar resultados
            self.model_results[model.model_name] = results
            
            logger.info(f"Modelo {model.model_name} evaluado exitosamente")
            return results
            
        except Exception as e:
            logger.error(f"Error evaluando modelo {model.model_name}: {str(e)}")
            raise
    
    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series,
                           X_train: Optional[pd.DataFrame] = None,
                           y_train: Optional[pd.Series] = None) -> Dict[str, Dict[str, Any]]:
        """
        Evalúa todos los modelos entrenados.
        
        Args:
            X_test: Features de prueba
            y_test: Target de prueba
            X_train: Features de entrenamiento (opcional, para PSI)
            y_train: Target de entrenamiento (opcional, para PSI)
            
        Returns:
            Diccionario con resultados de evaluación por modelo
        """
        logger.info("Evaluando todos los modelos")
        
        all_results = {}
        
        for model_name, model in self.models.items():
            if model.is_fitted:
                try:
                    logger.info(f"Evaluando {model_name}...")
                    results = self.evaluate_model(model, X_test, y_test, X_train, y_train)
                    all_results[model_name] = results
                    logger.info(f"✅ {model_name} evaluado exitosamente")
                    
                except Exception as e:
                    logger.error(f"❌ Error evaluando {model_name}: {str(e)}")
                    continue
            else:
                logger.warning(f"Modelo {model_name} no está entrenado")
        
        logger.info(f"Evaluación completada: {len(all_results)} modelos evaluados")
        
        return all_results
    
    def cross_validate_model(self, model: BaseCreditScoringModel,
                           X: pd.DataFrame, y: pd.Series,
                           scoring: str = 'roc_auc', cv: Optional[int] = None) -> Dict[str, Any]:
        """
        Realiza validación cruzada de un modelo.
        
        Args:
            model: Modelo a validar
            X: Features
            y: Target
            scoring: Métrica de evaluación
            cv: Número de folds (opcional)
            
        Returns:
            Resultados de validación cruzada
        """
        try:
            logger.info(f"Validación cruzada de {model.model_name}")
            results = model.cross_validate(X, y, scoring, cv)
            
            logger.info(f"CV completado para {model.model_name}: {scoring} = {results['mean_score']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error en CV de {model.model_name}: {str(e)}")
            raise
    
    def cross_validate_all_models(self, X: pd.DataFrame, y: pd.Series,
                                 scoring: str = 'roc_auc', cv: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """
        Realiza validación cruzada de todos los modelos.
        
        Args:
            X: Features
            y: Target
            scoring: Métrica de evaluación
            cv: Número de folds (opcional)
            
        Returns:
            Diccionario con resultados de CV por modelo
        """
        logger.info("Validación cruzada de todos los modelos")
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"CV de {model_name}...")
                results = self.cross_validate_model(model, X, y, scoring, cv)
                cv_results[model_name] = results
                logger.info(f"✅ CV de {model_name} completado")
                
            except Exception as e:
                logger.error(f"❌ Error en CV de {model_name}: {str(e)}")
                continue
        
        logger.info(f"CV completado: {len(cv_results)} modelos")
        
        return cv_results
    
    def get_best_model(self, metric: str = 'overall_score') -> Optional[BaseCreditScoringModel]:
        """
        Obtiene el mejor modelo según una métrica específica.
        
        Args:
            metric: Métrica para comparar ('overall_score', 'auc_roc', etc.)
            
        Returns:
            Mejor modelo o None si no hay modelos evaluados
        """
        if not self.model_results:
            logger.warning("No hay resultados de evaluación disponibles")
            return None
        
        best_model_name = None
        best_score = -1
        
        for model_name, results in self.model_results.items():
            try:
                if metric == 'overall_score':
                    score = results.get('overall_score', 0)
                elif metric in results.get('basic_metrics', {}):
                    score = results['basic_metrics'][metric]
                elif metric in results.get('credit_metrics', {}):
                    score = results['credit_metrics'][metric]
                else:
                    logger.warning(f"Métrica {metric} no encontrada en resultados")
                    continue
                
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    
            except Exception as e:
                logger.warning(f"Error obteniendo score de {model_name}: {e}")
                continue
        
        if best_model_name and best_model_name in self.models:
            logger.info(f"Mejor modelo: {best_model_name} (score: {best_score:.3f})")
            return self.models[best_model_name]
        
        return None
    
    def get_model_ranking(self, metric: str = 'overall_score') -> List[Dict[str, Any]]:
        """
        Obtiene ranking de modelos según una métrica específica.
        
        Args:
            metric: Métrica para comparar
            
        Returns:
            Lista de modelos ordenados por score
        """
        if not self.model_results:
            return []
        
        ranking = []
        
        for model_name, results in self.model_results.items():
            try:
                if metric == 'overall_score':
                    score = results.get('overall_score', 0)
                elif metric in results.get('basic_metrics', {}):
                    score = results['basic_metrics'][metric]
                elif metric in results.get('credit_metrics', {}):
                    score = results['credit_metrics'][metric]
                else:
                    continue
                
                ranking.append({
                    'model_name': model_name,
                    'score': score,
                    'metric': metric
                })
                
            except Exception as e:
                logger.warning(f"Error obteniendo score de {model_name}: {e}")
                continue
        
        # Ordenar por score descendente
        ranking.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Ranking generado para métrica {metric}: {len(ranking)} modelos")
        
        return ranking
    
    def get_model_info(self, model_name: str = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Obtiene información de un modelo específico o de todos.
        
        Args:
            model_name: Nombre del modelo (opcional)
            
        Returns:
            Información del modelo o de todos los modelos
        """
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Modelo {model_name} no encontrado")
            
            return self.models[model_name].get_model_info()
        
        else:
            all_info = {}
            for name, model in self.models.items():
                all_info[name] = model.get_model_info()
            
            return all_info
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """
        Obtiene lista de modelos disponibles.
        
        Returns:
            Diccionario con modelos disponibles
        """
        return {
            'all_models': CreditScoringModels.get_available_models(),
            'active_models': self.active_models,
            'created_models': list(self.models.keys()),
            'trained_models': [name for name, model in self.models.items() if model.is_fitted]
        }
    
    def save_models(self, base_path: str) -> None:
        """
        Guarda todos los modelos entrenados.
        
        Args:
            base_path: Ruta base donde guardar
        """
        from pathlib import Path
        
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Guardando modelos en: {base_path}")
        
        for model_name, model in self.models.items():
            if model.is_fitted:
                file_path = base_path / f"{model_name}_model.pkl"
                model.save_model(str(file_path))
                logger.info(f"  {model_name} guardado en {file_path}")
    
    def load_models(self, base_path: str) -> None:
        """
        Carga modelos desde archivos.
        
        Args:
            base_path: Ruta base donde están los archivos
        """
        from pathlib import Path
        
        base_path = Path(base_path)
        
        if not base_path.exists():
            logger.warning(f"Directorio {base_path} no existe")
            return
        
        logger.info(f"Cargando modelos desde: {base_path}")
        
        # Buscar archivos de modelos
        model_files = list(base_path.glob("*_model.pkl"))
        
        for file_path in model_files:
            try:
                model_name = file_path.stem.replace("_model", "")
                
                # Crear modelo temporal para cargar
                temp_model = self.create_model(model_name)
                temp_model.load_model(str(file_path))
                
                self.models[model_name] = temp_model
                logger.info(f"  {model_name} cargado desde {file_path}")
                
            except Exception as e:
                logger.warning(f"No se pudo cargar modelo desde {file_path}: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del entrenamiento.
        
        Returns:
            Resumen del entrenamiento
        """
        summary = {
            'total_models': len(self.models),
            'trained_models': sum(1 for m in self.models.values() if m.is_fitted),
            'evaluated_models': len(self.model_results),
            'model_types': {},
            'model_status': {}
        }
        
        for name, model in self.models.items():
            info = model.get_model_info()
            model_type = info.get('type', 'Unknown')
            
            if model_type not in summary['model_types']:
                summary['model_types'][model_type] = 0
            summary['model_types'][model_type] += 1
            
            summary['model_status'][name] = {
                'type': model_type,
                'is_fitted': model.is_fitted,
                'feature_count': info.get('feature_count', 0),
                'supports_probability': info.get('supports_probability', False)
            }
        
        return summary
    
    def __str__(self) -> str:
        """Representación string del factory."""
        summary = self.get_training_summary()
        return f"ModelFactory({summary['trained_models']}/{summary['total_models']} trained)"
    
    def __repr__(self) -> str:
        """Representación detallada del factory."""
        summary = self.get_training_summary()
        return f"ModelFactory(models={summary['total_models']}, trained={summary['trained_models']}, evaluated={summary['evaluated_models']})"
