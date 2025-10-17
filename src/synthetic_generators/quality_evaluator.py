"""
Evaluador de calidad para datos sintéticos.
Implementa las métricas definidas en el proyecto: KS, Chi-Squared, KL, Coseno, JSD.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.metrics import mutual_info_score
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class SyntheticQualityEvaluator:
    """
    Evaluador de calidad para datos sintéticos.
    Implementa las métricas definidas en el proyecto de grado.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el evaluador de calidad.
        
        Args:
            config: Configuración del proyecto
        """
        self.config = config
        self.evaluation_results = {}
        
        # Métricas a evaluar
        self.metrics = config.get('evaluation', {}).get('synthetic_quality', [
            'KS_complement',
            'chi_squared', 
            'kl_divergence_inverse',
            'cosine_similarity',
            'jensen_shannon_entropy'
        ])
        
        logger.info(f"Evaluador de calidad inicializado con {len(self.metrics)} métricas")
    
    def evaluate_quality(self, real_data: pd.DataFrame, 
                        synthetic_data: pd.DataFrame,
                        target_col: str = None) -> Dict[str, Any]:
        """
        Evalúa la calidad de los datos sintéticos.
        
        Args:
            real_data: Datos reales
            synthetic_data: Datos sintéticos
            target_col: Columna target (opcional)
            
        Returns:
            Diccionario con resultados de evaluación
        """
        logger.info("Iniciando evaluación de calidad de datos sintéticos")
        
        results = {
            'basic_comparison': self._basic_comparison(real_data, synthetic_data),
            'distribution_metrics': self._evaluate_distributions(real_data, synthetic_data),
            'correlation_metrics': self._evaluate_correlations(real_data, synthetic_data),
            'target_metrics': self._evaluate_target(real_data, synthetic_data, target_col),
            'overall_score': 0.0
        }
        
        # Calcular score general
        results['overall_score'] = self._calculate_overall_score(results)
        
        self.evaluation_results = results
        logger.info(f"Evaluación completada. Score general: {results['overall_score']:.3f}")
        
        return results
    
    def _basic_comparison(self, real_data: pd.DataFrame, 
                         synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Comparación básica entre datasets."""
        comparison = {
            'real_shape': real_data.shape,
            'synthetic_shape': synthetic_data.shape,
            'shape_match': real_data.shape[1] == synthetic_data.shape[1],
            'size_ratio': synthetic_data.shape[0] / real_data.shape[0] if real_data.shape[0] > 0 else 0,
            'column_match': set(real_data.columns) == set(synthetic_data.columns)
        }
        
        logger.info(f"Comparación básica: Real {comparison['real_shape']}, Synthetic {comparison['synthetic_shape']}")
        return comparison
    
    def _evaluate_distributions(self, real_data: pd.DataFrame, 
                               synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Evalúa distribuciones usando las métricas definidas."""
        numeric_cols = real_data.select_dtypes(include=[np.number]).columns
        distribution_metrics = {}
        
        for col in numeric_cols:
            if col in synthetic_data.columns:
                real_col = real_data[col].dropna()
                synthetic_col = synthetic_data[col].dropna()
                
                if len(real_col) > 0 and len(synthetic_col) > 0:
                    col_metrics = {}
                    
                    # 1. Kolmogorov-Smirnov Complement
                    if 'KS_complement' in self.metrics:
                        ks_stat, _ = stats.ks_2samp(real_col, synthetic_col)
                        col_metrics['KS_complement'] = 1 - ks_stat
                    
                    # 2. Chi-Squared Test
                    if 'chi_squared' in self.metrics:
                        chi2_stat = self._chi_squared_test(real_col, synthetic_col)
                        col_metrics['chi_squared'] = chi2_stat
                    
                    # 3. Kullback-Leibler Divergence Inverse
                    if 'kl_divergence_inverse' in self.metrics:
                        kl_div = self._kl_divergence_inverse(real_col, synthetic_col)
                        col_metrics['kl_divergence_inverse'] = kl_div
                    
                    # 4. Cosine Similarity
                    if 'cosine_similarity' in self.metrics:
                        cos_sim = self._cosine_similarity(real_col, synthetic_col)
                        col_metrics['cosine_similarity'] = cos_sim
                    
                    # 5. Jensen-Shannon Entropy
                    if 'jensen_shannon_entropy' in self.metrics:
                        js_entropy = self._jensen_shannon_entropy(real_col, synthetic_col)
                        col_metrics['jensen_shannon_entropy'] = js_entropy
                    
                    distribution_metrics[col] = col_metrics
        
        logger.info(f"Métricas de distribución calculadas para {len(distribution_metrics)} columnas")
        return distribution_metrics
    
    def _chi_squared_test(self, real_col: pd.Series, synthetic_col: pd.Series) -> float:
        """Calcula Chi-Squared test entre dos distribuciones."""
        try:
            # Crear bins para discretizar
            min_val = min(real_col.min(), synthetic_col.min())
            max_val = max(real_col.max(), synthetic_col.max())
            
            bins = np.linspace(min_val, max_val, 10)
            
            # Calcular frecuencias
            real_hist, _ = np.histogram(real_col, bins=bins)
            synthetic_hist, _ = np.histogram(synthetic_col, bins=bins)
            
            # Chi-squared test
            chi2_stat, p_value = stats.chisquare(real_hist, synthetic_hist)
            
            # Normalizar el estadístico
            return 1 / (1 + chi2_stat / len(real_col))
            
        except Exception as e:
            logger.warning(f"Error en chi-squared test: {e}")
            return 0.0
    
    def _kl_divergence_inverse(self, real_col: pd.Series, synthetic_col: pd.Series) -> float:
        """Calcula Kullback-Leibler Divergence inversa."""
        try:
            # Crear bins para discretizar
            min_val = min(real_col.min(), synthetic_col.min())
            max_val = max(real_col.max(), synthetic_col.max())
            
            bins = np.linspace(min_val, max_val, 20)
            
            # Calcular probabilidades
            real_hist, _ = np.histogram(real_col, bins=bins, density=True)
            synthetic_hist, _ = np.histogram(synthetic_col, bins=bins, density=True)
            
            # Evitar división por cero
            real_hist = real_hist + 1e-10
            synthetic_hist = synthetic_hist + 1e-10
            
            # Normalizar
            real_hist = real_hist / real_hist.sum()
            synthetic_hist = synthetic_hist / synthetic_hist.sum()
            
            # KL Divergence
            kl_div = np.sum(real_hist * np.log(real_hist / synthetic_hist))
            
            # KL Divergence inversa (IS Divergence)
            is_div = np.sum((real_hist / synthetic_hist) - np.log(real_hist / synthetic_hist) - 1)
            
            # Normalizar
            return 1 / (1 + is_div)
            
        except Exception as e:
            logger.warning(f"Error en KL divergence: {e}")
            return 0.0
    
    def _cosine_similarity(self, real_col: pd.Series, synthetic_col: pd.Series) -> float:
        """Calcula similitud del coseno entre distribuciones."""
        try:
            # Crear bins para discretizar
            min_val = min(real_col.min(), synthetic_col.min())
            max_val = max(real_col.max(), synthetic_col.max())
            
            bins = np.linspace(min_val, max_val, 20)
            
            # Calcular frecuencias
            real_hist, _ = np.histogram(real_col, bins=bins, density=True)
            synthetic_hist, _ = np.histogram(synthetic_col, bins=bins, density=True)
            
            # Calcular similitud del coseno
            cos_sim = 1 - cosine(real_hist, synthetic_hist)
            
            return max(0, cos_sim)  # Asegurar que sea no negativo
            
        except Exception as e:
            logger.warning(f"Error en cosine similarity: {e}")
            return 0.0
    
    def _jensen_shannon_entropy(self, real_col: pd.Series, synthetic_col: pd.Series) -> float:
        """Calcula entropía de Jensen-Shannon."""
        try:
            # Crear bins para discretizar
            min_val = min(real_col.min(), synthetic_col.min())
            max_val = max(real_col.max(), synthetic_col.max())
            
            bins = np.linspace(min_val, max_val, 20)
            
            # Calcular probabilidades
            real_hist, _ = np.histogram(real_col, bins=bins, density=True)
            synthetic_hist, _ = np.histogram(synthetic_col, bins=bins, density=True)
            
            # Evitar división por cero
            real_hist = real_hist + 1e-10
            synthetic_hist = synthetic_hist + 1e-10
            
            # Normalizar
            real_hist = real_hist / real_hist.sum()
            synthetic_hist = synthetic_hist / synthetic_hist.sum()
            
            # Calcular entropías
            def entropy(p):
                return -np.sum(p * np.log(p))
            
            # Jensen-Shannon Divergence
            m = (real_hist + synthetic_hist) / 2
            js_div = entropy(m) - 0.5 * (entropy(real_hist) + entropy(synthetic_hist))
            
            # Normalizar (JS divergence está entre 0 y 1)
            return 1 - js_div
            
        except Exception as e:
            logger.warning(f"Error en Jensen-Shannon entropy: {e}")
            return 0.0
    
    def _evaluate_correlations(self, real_data: pd.DataFrame, 
                              synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Evalúa correlaciones entre variables."""
        numeric_cols = real_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'error': 'No hay suficientes columnas numéricas para evaluar correlaciones'}
        
        try:
            # Calcular matrices de correlación
            real_corr = real_data[numeric_cols].corr()
            synthetic_corr = synthetic_data[numeric_cols].corr()
            
            # Calcular diferencia promedio
            corr_diff = abs(real_corr - synthetic_corr)
            avg_corr_diff = corr_diff.mean().mean()
            
            # Calcular correlación entre matrices
            real_corr_flat = real_corr.values[np.triu_indices_from(real_corr.values, k=1)]
            synthetic_corr_flat = synthetic_corr.values[np.triu_indices_from(synthetic_corr.values, k=1)]
            
            corr_correlation = np.corrcoef(real_corr_flat, synthetic_corr_flat)[0, 1]
            
            correlation_metrics = {
                'average_correlation_difference': avg_corr_diff,
                'correlation_correlation': corr_correlation,
                'correlation_similarity': 'high' if avg_corr_diff < 0.1 else 'medium' if avg_corr_diff < 0.2 else 'low'
            }
            
            logger.info(f"Correlaciones evaluadas: diferencia promedio = {avg_corr_diff:.3f}")
            return correlation_metrics
            
        except Exception as e:
            logger.warning(f"Error evaluando correlaciones: {e}")
            return {'error': str(e)}
    
    def _evaluate_target(self, real_data: pd.DataFrame, 
                        synthetic_data: pd.DataFrame, 
                        target_col: str) -> Dict[str, Any]:
        """Evalúa la distribución del target."""
        if not target_col or target_col not in real_data.columns or target_col not in synthetic_data.columns:
            return {'error': 'Target column not found or not specified'}
        
        try:
            real_target = real_data[target_col]
            synthetic_target = synthetic_data[target_col]
            
            # Distribuciones
            real_dist = real_target.value_counts(normalize=True).sort_index()
            synthetic_dist = synthetic_target.value_counts(normalize=True).sort_index()
            
            # Calcular diferencia
            all_classes = set(real_dist.index) | set(synthetic_dist.index)
            total_diff = 0
            
            for class_val in all_classes:
                real_pct = real_dist.get(class_val, 0)
                synthetic_pct = synthetic_dist.get(class_val, 0)
                total_diff += abs(real_pct - synthetic_pct)
            
            target_metrics = {
                'real_distribution': real_dist.to_dict(),
                'synthetic_distribution': synthetic_dist.to_dict(),
                'distribution_difference': total_diff,
                'distribution_similarity': 'high' if total_diff < 0.1 else 'medium' if total_diff < 0.2 else 'low'
            }
            
            logger.info(f"Target evaluado: diferencia = {total_diff:.3f}")
            return target_metrics
            
        except Exception as e:
            logger.warning(f"Error evaluando target: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calcula score general de calidad."""
        try:
            scores = []
            
            # Score de distribuciones
            if 'distribution_metrics' in results:
                dist_scores = []
                for col, metrics in results['distribution_metrics'].items():
                    col_scores = [v for v in metrics.values() if isinstance(v, (int, float)) and not np.isnan(v)]
                    if col_scores:
                        dist_scores.append(np.mean(col_scores))
                
                if dist_scores:
                    scores.append(np.mean(dist_scores))
            
            # Score de correlaciones
            if 'correlation_metrics' in results and 'correlation_correlation' in results['correlation_metrics']:
                corr_score = results['correlation_metrics']['correlation_correlation']
                if not np.isnan(corr_score):
                    scores.append(corr_score)
            
            # Score del target
            if 'target_metrics' in results and 'distribution_difference' in results['target_metrics']:
                target_diff = results['target_metrics']['distribution_difference']
                target_score = 1 - target_diff  # Convertir diferencia a similitud
                if not np.isnan(target_score):
                    scores.append(target_score)
            
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
        
        report = "=== EVALUACIÓN DE CALIDAD DE DATOS SINTÉTICOS ===\n\n"
        
        # Score general
        overall_score = self.evaluation_results.get('overall_score', 0)
        report += f"SCORE GENERAL: {overall_score:.3f}\n\n"
        
        # Comparación básica
        if 'basic_comparison' in self.evaluation_results:
            basic = self.evaluation_results['basic_comparison']
            report += "1. COMPARACIÓN BÁSICA\n"
            report += f"   Shape match: {basic['shape_match']}\n"
            report += f"   Size ratio: {basic['size_ratio']:.2f}\n"
            report += f"   Column match: {basic['column_match']}\n\n"
        
        # Métricas de distribución
        if 'distribution_metrics' in self.evaluation_results:
            dist_metrics = self.evaluation_results['distribution_metrics']
            report += "2. MÉTRICAS DE DISTRIBUCIÓN\n"
            
            for col, metrics in dist_metrics.items():
                report += f"   {col}:\n"
                for metric, value in metrics.items():
                    report += f"     {metric}: {value:.3f}\n"
                report += "\n"
        
        # Correlaciones
        if 'correlation_metrics' in self.evaluation_results:
            corr_metrics = self.evaluation_results['correlation_metrics']
            if 'error' not in corr_metrics:
                report += "3. CORRELACIONES\n"
                report += f"   Average difference: {corr_metrics.get('average_correlation_difference', 0):.3f}\n"
                report += f"   Correlation: {corr_metrics.get('correlation_correlation', 0):.3f}\n\n"
        
        # Target
        if 'target_metrics' in self.evaluation_results:
            target_metrics = self.evaluation_results['target_metrics']
            if 'error' not in target_metrics:
                report += "4. TARGET\n"
                report += f"   Distribution difference: {target_metrics.get('distribution_difference', 0):.3f}\n\n"
        
        return report
    
    def save_results(self, file_path: str) -> None:
        """Guarda resultados de evaluación."""
        import json
        
        with open(file_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        logger.info(f"Resultados de evaluación guardados en: {file_path}")
