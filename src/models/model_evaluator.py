"""
Evaluador específico para modelos de scoring crediticio.
Implementa métricas específicas del dominio: AUC, PSI, Traffic Light, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class CreditModelEvaluator:
    """
    Evaluador específico para modelos de scoring crediticio.
    Implementa métricas del dominio financiero.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el evaluador de modelos de crédito.
        
        Args:
            config: Configuración del proyecto
        """
        self.config = config
        self.evaluation_results = {}
        
        # Configuración de métricas
        self.credit_metrics = config.get('evaluation', {}).get('credit_metrics', [
            'auc_roc', 'psi', 'traffic_light', 'gini_coefficient', 'population_stability'
        ])
        
        # Umbrales de evaluación
        self.thresholds = config.get('evaluation', {}).get('thresholds', {
            'auc_roc': {'minimum': 0.65, 'target': 0.75},
            'psi': {'maximum': 0.10, 'warning': 0.05},
            'traffic_light': {'minimum_green': 0.8}
        })
        
        logger.info(f"CreditModelEvaluator inicializado con {len(self.credit_metrics)} métricas")
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                      X_train: Optional[pd.DataFrame] = None, 
                      y_train: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Evalúa un modelo de scoring crediticio.
        
        Args:
            model: Modelo entrenado
            X_test: Features de prueba
            y_test: Target de prueba
            X_train: Features de entrenamiento (opcional, para PSI)
            y_train: Target de entrenamiento (opcional, para PSI)
            
        Returns:
            Diccionario con resultados de evaluación
        """
        logger.info("Iniciando evaluación de modelo de scoring crediticio")
        
        results = {
            'basic_metrics': self._evaluate_basic_metrics(model, X_test, y_test),
            'credit_metrics': self._evaluate_credit_metrics(model, X_test, y_test, X_train, y_train),
            'stability_metrics': self._evaluate_stability_metrics(model, X_test, y_test, X_train, y_train),
            'overall_score': 0.0
        }
        
        # Calcular score general
        results['overall_score'] = self._calculate_overall_score(results)
        
        self.evaluation_results = results
        logger.info(f"Evaluación completada. Score general: {results['overall_score']:.3f}")
        
        return results
    
    def _evaluate_basic_metrics(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evalúa métricas básicas de ML."""
        try:
            # Predicciones
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Si y_proba tiene 2 columnas, tomar la segunda (clase positiva)
            if y_proba.shape[1] == 2:
                y_proba_positive = y_proba[:, 1]
            else:
                y_proba_positive = y_proba
            
            # Métricas básicas
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            basic_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'auc_roc': roc_auc_score(y_test, y_proba_positive)
            }
            
            # AUC-PR
            try:
                precision, recall, _ = precision_recall_curve(y_test, y_proba_positive)
                basic_metrics['auc_pr'] = auc(recall, precision)
            except Exception as e:
                logger.warning(f"Error calculando AUC-PR: {e}")
                basic_metrics['auc_pr'] = 0.0
            
            logger.info(f"Métricas básicas calculadas: AUC-ROC = {basic_metrics['auc_roc']:.3f}")
            return basic_metrics
            
        except Exception as e:
            logger.error(f"Error evaluando métricas básicas: {e}")
            return {'error': str(e)}
    
    def _evaluate_credit_metrics(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                                X_train: Optional[pd.DataFrame] = None,
                                y_train: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Evalúa métricas específicas de crédito."""
        credit_metrics = {}
        
        try:
            # Predicciones
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                y_proba_positive = y_proba[:, 1]
            else:
                y_proba_positive = y_proba
            
            # 1. AUC-ROC (ya calculado en básicas, pero lo incluimos aquí también)
            if 'auc_roc' in self.credit_metrics:
                credit_metrics['auc_roc'] = roc_auc_score(y_test, y_proba_positive)
            
            # 2. Gini Coefficient
            if 'gini_coefficient' in self.credit_metrics:
                auc_score = credit_metrics.get('auc_roc', roc_auc_score(y_test, y_proba_positive))
                credit_metrics['gini_coefficient'] = 2 * auc_score - 1
            
            # 3. PSI (Population Stability Index)
            if 'psi' in self.credit_metrics and X_train is not None and y_train is not None:
                psi_score = self._calculate_psi(model, X_train, X_test)
                credit_metrics['psi'] = psi_score
            
            # 4. Traffic Light
            if 'traffic_light' in self.credit_metrics:
                traffic_light = self._calculate_traffic_light(y_test, y_proba_positive)
                credit_metrics['traffic_light'] = traffic_light
            
            # 5. Population Stability
            if 'population_stability' in self.credit_metrics and X_train is not None:
                pop_stability = self._calculate_population_stability(model, X_train, X_test)
                credit_metrics['population_stability'] = pop_stability
            
            logger.info(f"Métricas de crédito calculadas: {len(credit_metrics)} métricas")
            return credit_metrics
            
        except Exception as e:
            logger.error(f"Error evaluando métricas de crédito: {e}")
            return {'error': str(e)}
    
    def _calculate_psi(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame) -> float:
        """Calcula Population Stability Index (PSI)."""
        try:
            # Predicciones en train y test
            train_proba = model.predict_proba(X_train)
            test_proba = model.predict_proba(X_test)
            
            if train_proba.shape[1] == 2:
                train_scores = train_proba[:, 1]
                test_scores = test_proba[:, 1]
            else:
                train_scores = train_proba
                test_scores = test_proba
            
            # Crear bins
            min_score = min(train_scores.min(), test_scores.min())
            max_score = max(train_scores.max(), test_scores.max())
            
            bins = np.linspace(min_score, max_score, 11)  # 10 bins
            
            # Calcular distribuciones
            train_hist, _ = np.histogram(train_scores, bins=bins)
            test_hist, _ = np.histogram(test_scores, bins=bins)
            
            # Normalizar
            train_dist = train_hist / train_hist.sum()
            test_dist = test_hist / test_hist.sum()
            
            # Evitar división por cero
            train_dist = np.where(train_dist == 0, 1e-6, train_dist)
            test_dist = np.where(test_dist == 0, 1e-6, test_dist)
            
            # Calcular PSI
            psi = np.sum((test_dist - train_dist) * np.log(test_dist / train_dist))
            
            logger.info(f"PSI calculado: {psi:.4f}")
            return psi
            
        except Exception as e:
            logger.warning(f"Error calculando PSI: {e}")
            return 0.0
    
    def _calculate_traffic_light(self, y_true: pd.Series, y_proba: np.ndarray) -> Dict[str, Any]:
        """Calcula métrica Traffic Light."""
        try:
            # Crear grupos de riesgo basados en percentiles
            n_groups = 10
            group_edges = np.linspace(0, 1, n_groups + 1)
            
            # Asignar grupos de riesgo
            risk_groups = pd.cut(y_proba, bins=group_edges, labels=False, include_lowest=True)
            
            # Calcular métricas por grupo
            group_metrics = []
            green_groups = 0
            
            for group in range(n_groups):
                group_mask = risk_groups == group
                if group_mask.sum() == 0:
                    continue
                
                group_y_true = y_true[group_mask]
                group_y_proba = y_proba[group_mask]
                
                # Tasa de default observada
                observed_default_rate = group_y_true.mean()
                
                # Tasa de default predicha (promedio de probabilidades)
                predicted_default_rate = group_y_proba.mean()
                
                # Calcular precisión del grupo
                if predicted_default_rate > 0:
                    precision = 1 - abs(observed_default_rate - predicted_default_rate) / predicted_default_rate
                else:
                    precision = 0.0
                
                group_metrics.append({
                    'group': group,
                    'observed_rate': observed_default_rate,
                    'predicted_rate': predicted_default_rate,
                    'precision': precision,
                    'status': 'green' if precision > 0.8 else 'yellow' if precision > 0.6 else 'red'
                })
                
                if precision > 0.8:
                    green_groups += 1
            
            # Métricas generales
            total_groups = len(group_metrics)
            green_percentage = green_groups / total_groups if total_groups > 0 else 0
            
            traffic_light = {
                'total_groups': total_groups,
                'green_groups': green_groups,
                'green_percentage': green_percentage,
                'group_metrics': group_metrics,
                'status': 'green' if green_percentage >= 0.8 else 'yellow' if green_percentage >= 0.6 else 'red'
            }
            
            logger.info(f"Traffic Light calculado: {green_percentage:.1%} grupos verdes")
            return traffic_light
            
        except Exception as e:
            logger.warning(f"Error calculando Traffic Light: {e}")
            return {'error': str(e)}
    
    def _calculate_population_stability(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Dict[str, Any]:
        """Calcula estabilidad de la población."""
        try:
            # Predicciones
            train_proba = model.predict_proba(X_train)
            test_proba = model.predict_proba(X_test)
            
            if train_proba.shape[1] == 2:
                train_scores = train_proba[:, 1]
                test_scores = test_proba[:, 1]
            else:
                train_scores = train_proba
                test_scores = test_proba
            
            # Estadísticas descriptivas
            train_stats = {
                'mean': train_scores.mean(),
                'std': train_scores.std(),
                'min': train_scores.min(),
                'max': train_scores.max(),
                'median': np.median(train_scores)
            }
            
            test_stats = {
                'mean': test_scores.mean(),
                'std': test_scores.std(),
                'min': test_scores.min(),
                'max': test_scores.max(),
                'median': np.median(test_scores)
            }
            
            # Calcular diferencias
            stability_metrics = {
                'mean_difference': abs(test_stats['mean'] - train_stats['mean']),
                'std_difference': abs(test_stats['std'] - train_stats['std']),
                'median_difference': abs(test_stats['median'] - train_stats['median']),
                'ks_statistic': stats.ks_2samp(train_scores, test_scores)[0]
            }
            
            # Score de estabilidad (0-1, donde 1 es perfectamente estable)
            stability_score = 1 - min(1, stability_metrics['ks_statistic'])
            
            population_stability = {
                'train_stats': train_stats,
                'test_stats': test_stats,
                'stability_metrics': stability_metrics,
                'stability_score': stability_score,
                'status': 'stable' if stability_score > 0.8 else 'warning' if stability_score > 0.6 else 'unstable'
            }
            
            logger.info(f"Estabilidad de población calculada: score = {stability_score:.3f}")
            return population_stability
            
        except Exception as e:
            logger.warning(f"Error calculando estabilidad de población: {e}")
            return {'error': str(e)}
    
    def _evaluate_stability_metrics(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                                   X_train: Optional[pd.DataFrame] = None,
                                   y_train: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Evalúa métricas de estabilidad."""
        stability_metrics = {}
        
        if X_train is not None:
            # PSI ya calculado en credit_metrics
            if 'psi' in self.evaluation_results.get('credit_metrics', {}):
                psi_score = self.evaluation_results['credit_metrics']['psi']
                stability_metrics['psi_stability'] = 'stable' if psi_score < 0.1 else 'warning' if psi_score < 0.2 else 'unstable'
            
            # Population Stability ya calculado
            if 'population_stability' in self.evaluation_results.get('credit_metrics', {}):
                pop_stability = self.evaluation_results['credit_metrics']['population_stability']
                stability_metrics['population_stability'] = pop_stability.get('status', 'unknown')
        
        return stability_metrics
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calcula score general del modelo."""
        try:
            scores = []
            
            # Score de métricas básicas
            if 'basic_metrics' in results and 'error' not in results['basic_metrics']:
                basic = results['basic_metrics']
                if 'auc_roc' in basic:
                    scores.append(basic['auc_roc'])
                if 'f1_score' in basic:
                    scores.append(basic['f1_score'])
            
            # Score de métricas de crédito
            if 'credit_metrics' in results and 'error' not in results['credit_metrics']:
                credit = results['credit_metrics']
                if 'gini_coefficient' in credit:
                    scores.append((credit['gini_coefficient'] + 1) / 2)  # Normalizar Gini a 0-1
                if 'traffic_light' in credit and 'error' not in credit['traffic_light']:
                    scores.append(credit['traffic_light']['green_percentage'])
            
            # Score de estabilidad
            if 'stability_metrics' in results:
                stability = results['stability_metrics']
                if 'psi_stability' in stability:
                    psi_score = 1.0 if stability['psi_stability'] == 'stable' else 0.5 if stability['psi_stability'] == 'warning' else 0.0
                    scores.append(psi_score)
            
            # Calcular score general
            if scores:
                overall_score = np.mean(scores)
                return max(0, min(1, overall_score))  # Asegurar que esté entre 0 y 1
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Error calculando score general: {e}")
            return 0.0
    
    def generate_report(self) -> str:
        """Genera reporte de evaluación."""
        if not self.evaluation_results:
            return "No hay resultados de evaluación disponibles."
        
        report = "=== EVALUACIÓN DE MODELO DE SCORING CREDITICIO ===\n\n"
        
        # Score general
        overall_score = self.evaluation_results.get('overall_score', 0)
        report += f"SCORE GENERAL: {overall_score:.3f}\n\n"
        
        # Métricas básicas
        if 'basic_metrics' in self.evaluation_results:
            basic = self.evaluation_results['basic_metrics']
            if 'error' not in basic:
                report += "1. MÉTRICAS BÁSICAS\n"
                report += f"   AUC-ROC: {basic.get('auc_roc', 0):.3f}\n"
                report += f"   AUC-PR: {basic.get('auc_pr', 0):.3f}\n"
                report += f"   Precision: {basic.get('precision', 0):.3f}\n"
                report += f"   Recall: {basic.get('recall', 0):.3f}\n"
                report += f"   F1-Score: {basic.get('f1_score', 0):.3f}\n\n"
        
        # Métricas de crédito
        if 'credit_metrics' in self.evaluation_results:
            credit = self.evaluation_results['credit_metrics']
            if 'error' not in credit:
                report += "2. MÉTRICAS DE CRÉDITO\n"
                report += f"   Gini Coefficient: {credit.get('gini_coefficient', 0):.3f}\n"
                report += f"   PSI: {credit.get('psi', 0):.3f}\n"
                
                if 'traffic_light' in credit and 'error' not in credit['traffic_light']:
                    tl = credit['traffic_light']
                    report += f"   Traffic Light: {tl.get('green_percentage', 0):.1%} grupos verdes\n"
                
                report += "\n"
        
        # Estabilidad
        if 'stability_metrics' in self.evaluation_results:
            stability = self.evaluation_results['stability_metrics']
            report += "3. ESTABILIDAD\n"
            report += f"   PSI Status: {stability.get('psi_stability', 'unknown')}\n"
            report += f"   Population Stability: {stability.get('population_stability', 'unknown')}\n\n"
        
        return report
    
    def save_results(self, file_path: str) -> None:
        """Guarda resultados de evaluación."""
        import json
        
        with open(file_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        logger.info(f"Resultados de evaluación guardados en: {file_path}")
