"""
Validador de datos para el proyecto de grado.
Maneja la validación de calidad y consistencia de datos.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy import stats
from sklearn.metrics import mutual_info_score

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Clase para validar calidad y consistencia de datos.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el validador de datos.
        
        Args:
            config: Configuración del proyecto
        """
        self.config = config
        self.validation_results = {}
        
    def validate_data_quality(self, data: pd.DataFrame, 
                            target_col: str = None) -> Dict[str, Any]:
        """
        Valida la calidad general de los datos.
        
        Args:
            data: DataFrame a validar
            target_col: Columna target (opcional)
            
        Returns:
            Diccionario con resultados de validación
        """
        logger.info("Validando calidad de datos")
        
        results = {
            'basic_info': self._get_basic_info(data),
            'missing_values': self._check_missing_values(data),
            'duplicates': self._check_duplicates(data),
            'data_types': self._check_data_types(data),
            'outliers': self._check_outliers(data),
            'correlations': self._check_correlations(data, target_col)
        }
        
        self.validation_results['quality'] = results
        return results
    
    def _get_basic_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Obtiene información básica del dataset."""
        info = {
            'shape': data.shape,
            'memory_usage': data.memory_usage(deep=True).sum(),
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict()
        }
        
        logger.info(f"  Shape: {info['shape']}")
        logger.info(f"  Memory usage: {info['memory_usage'] / 1024**2:.2f} MB")
        
        return info
    
    def _check_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Verifica valores faltantes."""
        missing_info = {}
        
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            missing_pct = (missing_count / len(data)) * 100
            
            missing_info[col] = {
                'count': missing_count,
                'percentage': missing_pct,
                'status': 'ok' if missing_pct < 5 else 'warning' if missing_pct < 20 else 'critical'
            }
        
        # Resumen
        total_missing = data.isnull().sum().sum()
        total_pct = (total_missing / (len(data) * len(data.columns))) * 100
        
        missing_info['summary'] = {
            'total_missing': total_missing,
            'total_percentage': total_pct,
            'status': 'ok' if total_pct < 5 else 'warning' if total_pct < 20 else 'critical'
        }
        
        logger.info(f"  Missing values: {total_missing} ({total_pct:.2f}%)")
        
        return missing_info
    
    def _check_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Verifica duplicados."""
        duplicate_count = data.duplicated().sum()
        duplicate_pct = (duplicate_count / len(data)) * 100
        
        duplicate_info = {
            'count': duplicate_count,
            'percentage': duplicate_pct,
            'status': 'ok' if duplicate_pct < 1 else 'warning' if duplicate_pct < 5 else 'critical'
        }
        
        logger.info(f"  Duplicates: {duplicate_count} ({duplicate_pct:.2f}%)")
        
        return duplicate_info
    
    def _check_data_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Verifica tipos de datos."""
        type_info = {}
        
        for col in data.columns:
            dtype = data[col].dtype
            unique_count = data[col].nunique()
            unique_pct = (unique_count / len(data)) * 100
            
            type_info[col] = {
                'dtype': str(dtype),
                'unique_count': unique_count,
                'unique_percentage': unique_pct,
                'status': 'ok' if unique_pct > 1 else 'warning'
            }
        
        logger.info(f"  Data types checked for {len(data.columns)} columns")
        
        return type_info
    
    def _check_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Verifica outliers en variables numéricas."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(data)) * 100
            
            outlier_info[col] = {
                'count': outlier_count,
                'percentage': outlier_pct,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'status': 'ok' if outlier_pct < 5 else 'warning' if outlier_pct < 15 else 'critical'
            }
        
        logger.info(f"  Outliers checked for {len(numeric_cols)} numeric columns")
        
        return outlier_info
    
    def _check_correlations(self, data: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        """Verifica correlaciones."""
        correlation_info = {}
        
        # Correlaciones entre features
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            corr_matrix = numeric_data.corr()
            
            # Encontrar correlaciones altas
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            correlation_info['feature_correlations'] = {
                'high_correlation_pairs': high_corr_pairs,
                'max_correlation': corr_matrix.abs().max().max()
            }
        
        # Correlaciones con target
        if target_col and target_col in data.columns:
            target_correlations = {}
            for col in numeric_data.columns:
                if col != target_col:
                    corr_val = data[col].corr(data[target_col])
                    target_correlations[col] = corr_val
            
            correlation_info['target_correlations'] = target_correlations
        
        logger.info(f"  Correlations checked")
        
        return correlation_info
    
    def validate_synthetic_data_quality(self, real_data: pd.DataFrame, 
                                      synthetic_data: pd.DataFrame,
                                      target_col: str = None) -> Dict[str, Any]:
        """
        Valida la calidad de datos sintéticos comparándolos con datos reales.
        
        Args:
            real_data: Datos reales
            synthetic_data: Datos sintéticos
            target_col: Columna target
            
        Returns:
            Diccionario con resultados de validación
        """
        logger.info("Validando calidad de datos sintéticos")
        
        results = {
            'shape_comparison': self._compare_shapes(real_data, synthetic_data),
            'statistical_comparison': self._compare_statistics(real_data, synthetic_data),
            'distribution_comparison': self._compare_distributions(real_data, synthetic_data),
            'correlation_comparison': self._compare_correlations(real_data, synthetic_data),
            'target_comparison': self._compare_targets(real_data, synthetic_data, target_col)
        }
        
        self.validation_results['synthetic_quality'] = results
        return results
    
    def _compare_shapes(self, real_data: pd.DataFrame, 
                       synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Compara formas de los datasets."""
        comparison = {
            'real_shape': real_data.shape,
            'synthetic_shape': synthetic_data.shape,
            'shape_match': real_data.shape[1] == synthetic_data.shape[1],
            'size_ratio': synthetic_data.shape[0] / real_data.shape[0]
        }
        
        logger.info(f"  Shape comparison: Real {comparison['real_shape']}, Synthetic {comparison['synthetic_shape']}")
        
        return comparison
    
    def _compare_statistics(self, real_data: pd.DataFrame, 
                           synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Compara estadísticas descriptivas."""
        numeric_cols = real_data.select_dtypes(include=[np.number]).columns
        
        comparison = {}
        
        for col in numeric_cols:
            if col in synthetic_data.columns:
                real_stats = real_data[col].describe()
                synthetic_stats = synthetic_data[col].describe()
                
                comparison[col] = {
                    'real_mean': real_stats['mean'],
                    'synthetic_mean': synthetic_stats['mean'],
                    'mean_diff': abs(real_stats['mean'] - synthetic_stats['mean']),
                    'real_std': real_stats['std'],
                    'synthetic_std': synthetic_stats['std'],
                    'std_diff': abs(real_stats['std'] - synthetic_stats['std'])
                }
        
        logger.info(f"  Statistical comparison for {len(comparison)} numeric columns")
        
        return comparison
    
    def _compare_distributions(self, real_data: pd.DataFrame, 
                              synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Compara distribuciones usando KS test."""
        numeric_cols = real_data.select_dtypes(include=[np.number]).columns
        
        comparison = {}
        
        for col in numeric_cols:
            if col in synthetic_data.columns:
                # KS test
                ks_stat, ks_pvalue = stats.ks_2samp(real_data[col], synthetic_data[col])
                
                comparison[col] = {
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pvalue,
                    'ks_complement': 1 - ks_stat,
                    'distribution_similarity': 'high' if ks_stat < 0.1 else 'medium' if ks_stat < 0.2 else 'low'
                }
        
        logger.info(f"  Distribution comparison for {len(comparison)} numeric columns")
        
        return comparison
    
    def _compare_correlations(self, real_data: pd.DataFrame, 
                             synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Compara matrices de correlación."""
        numeric_cols = real_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            real_corr = real_data[numeric_cols].corr()
            synthetic_corr = synthetic_data[numeric_cols].corr()
            
            # Calcular diferencia promedio
            corr_diff = abs(real_corr - synthetic_corr)
            avg_corr_diff = corr_diff.mean().mean()
            
            comparison = {
                'average_correlation_difference': avg_corr_diff,
                'correlation_similarity': 'high' if avg_corr_diff < 0.1 else 'medium' if avg_corr_diff < 0.2 else 'low'
            }
        else:
            comparison = {'error': 'Not enough numeric columns for correlation comparison'}
        
        logger.info(f"  Correlation comparison completed")
        
        return comparison
    
    def _compare_targets(self, real_data: pd.DataFrame, 
                        synthetic_data: pd.DataFrame, 
                        target_col: str) -> Dict[str, Any]:
        """Compara distribuciones del target."""
        if not target_col or target_col not in real_data.columns or target_col not in synthetic_data.columns:
            return {'error': 'Target column not found or not specified'}
        
        real_target = real_data[target_col]
        synthetic_target = synthetic_data[target_col]
        
        comparison = {
            'real_distribution': real_target.value_counts(normalize=True).to_dict(),
            'synthetic_distribution': synthetic_target.value_counts(normalize=True).to_dict(),
            'distribution_difference': abs(real_target.value_counts(normalize=True) - 
                                         synthetic_target.value_counts(normalize=True)).sum()
        }
        
        logger.info(f"  Target comparison completed")
        
        return comparison
    
    def generate_validation_report(self) -> str:
        """
        Genera un reporte de validación.
        
        Returns:
            Reporte en formato string
        """
        report = "=== DATA VALIDATION REPORT ===\n\n"
        
        if 'quality' in self.validation_results:
            quality = self.validation_results['quality']
            
            report += "1. DATA QUALITY\n"
            report += f"   Shape: {quality['basic_info']['shape']}\n"
            report += f"   Memory usage: {quality['basic_info']['memory_usage'] / 1024**2:.2f} MB\n"
            
            # Missing values summary
            missing_summary = quality['missing_values']['summary']
            report += f"   Missing values: {missing_summary['total_missing']} ({missing_summary['total_percentage']:.2f}%)\n"
            
            # Duplicates
            duplicates = quality['duplicates']
            report += f"   Duplicates: {duplicates['count']} ({duplicates['percentage']:.2f}%)\n"
            
            report += "\n"
        
        if 'synthetic_quality' in self.validation_results:
            synthetic = self.validation_results['synthetic_quality']
            
            report += "2. SYNTHETIC DATA QUALITY\n"
            
            # Shape comparison
            shape_comp = synthetic['shape_comparison']
            report += f"   Shape match: {shape_comp['shape_match']}\n"
            report += f"   Size ratio: {shape_comp['size_ratio']:.2f}\n"
            
            # Distribution comparison
            dist_comp = synthetic['distribution_comparison']
            if dist_comp:
                avg_ks = np.mean([v['ks_statistic'] for v in dist_comp.values()])
                report += f"   Average KS statistic: {avg_ks:.3f}\n"
            
            report += "\n"
        
        return report
    
    def save_validation_results(self, file_path: str) -> None:
        """
        Guarda resultados de validación.
        
        Args:
            file_path: Ruta donde guardar
        """
        import json
        
        with open(file_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        logger.info(f"Validation results saved to: {file_path}")
