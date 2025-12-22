"""
Exploratory Data Analysis module for DataAnalyzer 2.0
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import time

def descriptive_statistics(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                          params: Dict = {}) -> Dict:
    """
    Calcule les statistiques descriptives
    
    Args:
        df: DataFrame
        columns: Liste des colonnes à analyser (None = toutes numériques)
        params: Paramètres additionnels
        
    Returns:
        Dictionnaire des résultats
    """
    start_time = time.time()
    
    try:
        # Sélectionner les colonnes numériques
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in columns if col in df.select_dtypes(include=[np.number]).columns]
        
        if not numeric_cols:
            return {
                'success': False,
                'results': {},
                'error': "Aucune colonne numérique trouvée"
            }
        
        # Calculer les statistiques
        stats_df = df[numeric_cols].describe().T
        
        # Ajouter des statistiques supplémentaires
        stats_df['skewness'] = df[numeric_cols].skew()
        stats_df['kurtosis'] = df[numeric_cols].kurtosis()
        stats_df['missing'] = df[numeric_cols].isna().sum()
        stats_df['missing_pct'] = (df[numeric_cols].isna().sum() / len(df)) * 100
        
        results = {
            'success': True,
            'results': {
                'statistics': stats_df.to_dict('index'),
                'columns_analyzed': numeric_cols,
                'n_columns': len(numeric_cols)
            },
            'visualizations': [],
            'explanations': {
                'method': "Statistiques descriptives calculées pour toutes les variables numériques",
                'interpretation': """
                - Mean/Std: Moyenne et dispersion
                - Min/Max: Valeurs extrêmes
                - 25%/50%/75%: Quartiles (médiane = 50%)
                - Skewness: Asymétrie (>0 = étalée à droite, <0 = étalée à gauche)
                - Kurtosis: Aplatissement (>0 = queue épaisse, <0 = queue légère)
                """,
                'warnings': []
            },
            'execution_time': time.time() - start_time
        }
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'results': {},
            'error': str(e),
            'execution_time': time.time() - start_time
        }

def correlation_analysis(df: pd.DataFrame, columns: Optional[List[str]] = None,
                        params: Dict = {}) -> Dict:
    """
    Analyse de corrélation entre variables numériques
    
    Args:
        df: DataFrame
        columns: Liste des colonnes à analyser
        params: Paramètres (method='pearson'|'spearman', threshold=0.0)
        
    Returns:
        Dictionnaire des résultats
    """
    start_time = time.time()
    
    try:
        method = params.get('method', 'pearson')
        threshold = params.get('threshold', 0.0)
        
        # Sélectionner les colonnes numériques
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in columns if col in df.select_dtypes(include=[np.number]).columns]
        
        if len(numeric_cols) < 2:
            return {
                'success': False,
                'results': {},
                'error': "Au moins 2 colonnes numériques nécessaires"
            }
        
        # Calculer la corrélation
        corr_matrix = df[numeric_cols].corr(method=method)
        
        # Trouver les corrélations fortes
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': round(corr_value, 3),
                        'strength': 'Forte' if abs(corr_value) >= 0.7 else 'Modérée'
                    })
        
        # Trier par valeur absolue de corrélation
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        results = {
            'success': True,
            'results': {
                'correlation_matrix': corr_matrix.to_dict(),
                'strong_correlations': strong_correlations,
                'method': method,
                'threshold': threshold,
                'n_pairs': len(strong_correlations)
            },
            'visualizations': ['correlation_heatmap'],
            'explanations': {
                'method': f"Corrélation {method} calculée entre toutes les paires de variables numériques",
                'interpretation': """
                - Corrélation > 0.7: Forte corrélation positive
                - Corrélation < -0.7: Forte corrélation négative
                - Corrélation proche de 0: Pas de corrélation linéaire
                - Pearson: Relations linéaires
                - Spearman: Relations monotones (robuste aux outliers)
                """,
                'warnings': ['Corrélation ≠ Causalité']
            },
            'execution_time': time.time() - start_time
        }
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'results': {},
            'error': str(e),
            'execution_time': time.time() - start_time
        }

def detect_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None,
                   params: Dict = {}) -> Dict:
    """
    Détecte les outliers avec la méthode IQR
    
    Args:
        df: DataFrame
        columns: Liste des colonnes à analyser
        params: Paramètres (iqr_multiplier=1.5)
        
    Returns:
        Dictionnaire des résultats
    """
    start_time = time.time()
    
    try:
        iqr_multiplier = params.get('iqr_multiplier', 1.5)
        
        # Sélectionner les colonnes numériques
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in columns if col in df.select_dtypes(include=[np.number]).columns]
        
        if not numeric_cols:
            return {
                'success': False,
                'results': {},
                'error': "Aucune colonne numérique trouvée"
            }
        
        outliers_summary = {}
        all_outliers_indices = set()
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            
            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers_indices = df[outliers_mask].index.tolist()
            all_outliers_indices.update(outliers_indices)
            
            outliers_summary[col] = {
                'n_outliers': outliers_mask.sum(),
                'pct_outliers': (outliers_mask.sum() / len(df)) * 100,
                'lower_bound': round(lower_bound, 3),
                'upper_bound': round(upper_bound, 3),
                'outlier_values': df.loc[outliers_mask, col].tolist()[:10]  # Max 10 exemples
            }
        
        results = {
            'success': True,
            'results': {
                'outliers_by_column': outliers_summary,
                'total_outlier_rows': len(all_outliers_indices),
                'pct_outlier_rows': (len(all_outliers_indices) / len(df)) * 100,
                'iqr_multiplier': iqr_multiplier
            },
            'visualizations': ['boxplots'],
            'explanations': {
                'method': f"Détection par méthode IQR (multiplicateur = {iqr_multiplier})",
                'interpretation': """
                Un outlier est une valeur en dehors de [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
                - Q1: 25ème percentile
                - Q3: 75ème percentile
                - IQR: Écart interquartile (Q3 - Q1)
                """,
                'warnings': [
                    'Ne pas supprimer automatiquement les outliers',
                    'Comprendre leur origine (erreur ou valeur réelle extrême)'
                ]
            },
            'execution_time': time.time() - start_time
        }
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'results': {},
            'error': str(e),
            'execution_time': time.time() - start_time
        }

def categorical_analysis(df: pd.DataFrame, columns: Optional[List[str]] = None,
                        params: Dict = {}) -> Dict:
    """
    Analyse des variables catégorielles
    
    Args:
        df: DataFrame
        columns: Liste des colonnes à analyser
        params: Paramètres additionnels
        
    Returns:
        Dictionnaire des résultats
    """
    start_time = time.time()
    
    try:
        # Sélectionner les colonnes catégorielles
        if columns is None:
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            cat_cols = [col for col in columns if col in df.select_dtypes(include=['object', 'category']).columns]
        
        if not cat_cols:
            return {
                'success': False,
                'results': {},
                'error': "Aucune colonne catégorielle trouvée"
            }
        
        categorical_summary = {}
        
        for col in cat_cols:
            value_counts = df[col].value_counts()
            n_unique = df[col].nunique()
            
            # Calculer l'entropie
            probabilities = value_counts / len(df)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            max_entropy = np.log2(n_unique) if n_unique > 0 else 0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            categorical_summary[col] = {
                'n_unique': n_unique,
                'top_values': value_counts.head(10).to_dict(),
                'entropy': round(entropy, 3),
                'normalized_entropy': round(normalized_entropy, 3),
                'missing': df[col].isna().sum(),
                'missing_pct': (df[col].isna().sum() / len(df)) * 100
            }
        
        results = {
            'success': True,
            'results': {
                'categorical_summary': categorical_summary,
                'columns_analyzed': cat_cols,
                'n_columns': len(cat_cols)
            },
            'visualizations': ['bar_charts'],
            'explanations': {
                'method': "Analyse de fréquence et entropie pour variables catégorielles",
                'interpretation': """
                - N_unique: Nombre de catégories différentes
                - Entropy: Mesure du désordre (0 = une seule catégorie, max = distribution uniforme)
                - Top_values: Catégories les plus fréquentes
                """,
                'warnings': [
                    'Haute cardinalité (>50 catégories) peut poser problème',
                    'Considérer le regroupement des catégories rares'
                ]
            },
            'execution_time': time.time() - start_time
        }
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'results': {},
            'error': str(e),
            'execution_time': time.time() - start_time
        }

def distribution_analysis(df: pd.DataFrame, columns: Optional[List[str]] = None,
                         params: Dict = {}) -> Dict:
    """
    Analyse de la distribution des variables numériques
    
    Args:
        df: DataFrame
        columns: Liste des colonnes à analyser
        params: Paramètres (bins=30)
        
    Returns:
        Dictionnaire des résultats
    """
    start_time = time.time()
    
    try:
        bins = params.get('bins', 30)
        
        # Sélectionner les colonnes numériques
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in columns if col in df.select_dtypes(include=[np.number]).columns]
        
        if not numeric_cols:
            return {
                'success': False,
                'results': {},
                'error': "Aucune colonne numérique trouvée"
            }
        
        distribution_summary = {}
        
        for col in numeric_cols:
            data_clean = df[col].dropna()
            
            # Test de normalité (Shapiro-Wilk pour petits échantillons)
            if len(data_clean) < 5000:
                _, p_value = stats.shapiro(data_clean.sample(min(5000, len(data_clean))))
                is_normal = p_value > 0.05
            else:
                is_normal = None  # Trop grand pour Shapiro
            
            distribution_summary[col] = {
                'mean': round(data_clean.mean(), 3),
                'median': round(data_clean.median(), 3),
                'mode': data_clean.mode().tolist()[0] if not data_clean.mode().empty else None,
                'std': round(data_clean.std(), 3),
                'skewness': round(data_clean.skew(), 3),
                'kurtosis': round(data_clean.kurtosis(), 3),
                'is_normal': is_normal,
                'range': [round(data_clean.min(), 3), round(data_clean.max(), 3)]
            }
        
        results = {
            'success': True,
            'results': {
                'distribution_summary': distribution_summary,
                'columns_analyzed': numeric_cols,
                'bins': bins
            },
            'visualizations': ['histograms'],
            'explanations': {
                'method': "Analyse de la forme et des caractéristiques des distributions",
                'interpretation': """
                - Skewness > 0: Distribution étalée à droite
                - Skewness < 0: Distribution étalée à gauche
                - Kurtosis > 0: Queue épaisse (plus d'outliers)
                - is_normal: Test de normalité (Shapiro-Wilk, p > 0.05)
                """,
                'warnings': [
                    'Une distribution non-normale peut nécessiter des transformations',
                    'Vérifier la présence d\'outliers'
                ]
            },
            'execution_time': time.time() - start_time
        }
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'results': {},
            'error': str(e),
            'execution_time': time.time() - start_time
        }
