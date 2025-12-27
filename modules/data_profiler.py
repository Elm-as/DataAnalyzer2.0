"""
Data profiling module for DataAnalyzer 2.0
Automatic type detection and quality metrics
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def detect_column_type(series: pd.Series) -> str:
    """
    Détecte automatiquement le type d'une colonne
    
    Args:
        series: pandas Series à analyser
        
    Returns:
        Type détecté: 'numeric', 'categorical', 'text', 'date', 'boolean'
    """
    # Enlever les valeurs manquantes pour l'analyse
    series_clean = series.dropna()
    
    if len(series_clean) == 0:
        return 'unknown'
    
    # Vérifier si c'est une date
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'date'
    
    # Essayer de convertir en datetime
    if series.dtype == 'object':
        try:
            pd.to_datetime(series_clean.head(100))
            return 'date'
        except:
            pass
    
    # Vérifier si c'est booléen
    if pd.api.types.is_bool_dtype(series):
        return 'boolean'
    
    # Vérifier si les valeurs uniques sont seulement True/False ou 0/1 (boolean detection)
    unique_values = set(series_clean.unique())
    # Classify as boolean if exactly 2 unique boolean-like values (including numeric 0/1)
    if len(unique_values) == 2:
        # String boolean values
        if unique_values.issubset({True, False, 'True', 'False', 'true', 'false'}):
            return 'boolean'
        # Numeric boolean values (0 and 1 only)
        if unique_values.issubset({0, 1}):
            return 'boolean'
    
    # Vérifier si c'est numérique
    if pd.api.types.is_numeric_dtype(series):
        # Keep numeric columns as numeric, even if they have few unique values
        # Low cardinality numeric columns (like SibSp, Parch) should stay numeric
        # They represent counts/quantities, not categories
        return 'numeric'
    
    # Pour les colonnes object
    if series.dtype == 'object':
        # Essayer de convertir en numérique
        try:
            pd.to_numeric(series_clean)
            return 'numeric'
        except:
            pass
        
        # Vérifier si c'est catégoriel (peu de valeurs uniques)
        n_unique = series_clean.nunique()
        n_total = len(series_clean)
        
        # Catégoriel si < 5% de valeurs uniques ou < 50 valeurs uniques
        if n_unique < n_total * 0.05 or n_unique <= 50:
            return 'categorical'
        
        # Vérifier la longueur moyenne du texte
        avg_length = series_clean.astype(str).str.len().mean()
        if avg_length > 50:  # Texte long
            return 'text'
        
        return 'categorical'
    
    return 'unknown'

def profile_dataframe(df: pd.DataFrame) -> Dict:
    """
    Analyse complète d'un DataFrame
    
    Args:
        df: DataFrame à profiler
        
    Returns:
        Dictionnaire avec toutes les métriques de qualité
    """
    profile = {
        'shape': df.shape,
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'columns': {},
        'quality_metrics': {},
        'type_summary': {}
    }
    
    # Analyser chaque colonne
    for col in df.columns:
        col_type = detect_column_type(df[col])
        n_missing = df[col].isna().sum()
        n_unique = df[col].nunique()
        
        col_info = {
            'type': col_type,
            'dtype': str(df[col].dtype),
            'n_missing': n_missing,
            'pct_missing': (n_missing / len(df)) * 100,
            'n_unique': n_unique,
            'pct_unique': (n_unique / len(df)) * 100
        }
        
        # Ajouter des stats spécifiques selon le type
        if col_type == 'numeric':
            col_info.update({
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std()
            })
        elif col_type == 'categorical':
            value_counts = df[col].value_counts()
            col_info.update({
                'top_values': value_counts.head(10).to_dict(),
                'n_categories': n_unique
            })
        
        profile['columns'][col] = col_info
    
    # Métriques globales de qualité
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isna().sum().sum()
    
    profile['quality_metrics'] = {
        'total_missing': total_missing,
        'pct_missing': (total_missing / total_cells) * 100,
        'n_duplicates': df.duplicated().sum(),
        'pct_duplicates': (df.duplicated().sum() / len(df)) * 100,
        'complete_rows': df.dropna().shape[0],
        'pct_complete_rows': (df.dropna().shape[0] / len(df)) * 100
    }
    
    # Résumé des types
    type_counts = {}
    for col_info in profile['columns'].values():
        col_type = col_info['type']
        type_counts[col_type] = type_counts.get(col_type, 0) + 1
    
    profile['type_summary'] = type_counts
    
    return profile

def get_column_recommendations(df: pd.DataFrame, profile: Dict) -> Dict[str, List[str]]:
    """
    Génère des recommandations pour le traitement des colonnes
    
    Args:
        df: DataFrame
        profile: Profil généré par profile_dataframe
        
    Returns:
        Dictionnaire des recommandations par colonne
    """
    recommendations = {}
    
    for col, col_info in profile['columns'].items():
        col_recs = []
        
        # Valeurs manquantes
        if col_info['pct_missing'] > 50:
            col_recs.append(f"Attention: {col_info['pct_missing']:.1f}% de valeurs manquantes - Considérer la suppression")
        elif col_info['pct_missing'] > 5:
            col_recs.append(f"Info: {col_info['pct_missing']:.1f}% de valeurs manquantes - Imputation recommandée")
        
        # Cardinalité élevée pour catégorielles
        if col_info['type'] == 'categorical' and col_info['n_unique'] > 50:
            col_recs.append(f"Attention: Haute cardinalité ({col_info['n_unique']} catégories) - Regroupement recommandé")
        
        # Variance nulle
        if col_info['type'] == 'numeric' and col_info.get('std', 1) == 0:
            col_recs.append("Attention: Variance nulle - Variable constante, peut être supprimée")
        
        # Une seule valeur
        if col_info['n_unique'] == 1:
            col_recs.append("Attention: Une seule valeur unique - Variable inutile")
        
        if col_recs:
            recommendations[col] = col_recs
    
    return recommendations

def suggest_target_variables(df: pd.DataFrame, profile: Dict) -> List[str]:
    """
    Suggère des variables cibles potentielles
    
    Args:
        df: DataFrame
        profile: Profil du DataFrame
        
    Returns:
        Liste des colonnes candidates pour être la cible
    """
    candidates = []
    
    for col, col_info in profile['columns'].items():
        # Critères pour être une bonne cible
        if col_info['pct_missing'] < 10:  # Peu de valeurs manquantes
            if col_info['type'] in ['numeric', 'categorical', 'boolean']:
                # Pas trop de catégories pour du catégoriel
                if col_info['type'] == 'categorical' and col_info['n_unique'] <= 20:
                    candidates.append(col)
                elif col_info['type'] in ['numeric', 'boolean']:
                    candidates.append(col)
    
    return candidates

def get_data_quality_score(profile: Dict) -> Tuple[float, str]:
    """
    Calcule un score de qualité des données (0-100)
    
    Args:
        profile: Profil du DataFrame
        
    Returns:
        (score, niveau de qualité)
    """
    quality_metrics = profile['quality_metrics']
    
    # Facteurs de qualité
    completeness_score = 100 - quality_metrics['pct_missing']
    uniqueness_score = 100 - quality_metrics['pct_duplicates']
    
    # Score global (moyenne pondérée)
    total_score = (completeness_score * 0.6 + uniqueness_score * 0.4)
    
    if total_score >= 90:
        quality_level = "Excellente"
    elif total_score >= 75:
        quality_level = "Bonne"
    elif total_score >= 50:
        quality_level = "Moyenne"
    else:
        quality_level = "Faible"
    
    return round(total_score, 1), quality_level
