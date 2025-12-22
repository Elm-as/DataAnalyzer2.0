"""
Scientific validation module for DataAnalyzer 2.0
Ensures proper separation of target and features
"""

def validate_target_not_in_features(features, target):
    """
    RÈGLE 1: La cible ne doit JAMAIS être dans les features
    
    Args:
        features: Liste des features sélectionnées
        target: Nom de la variable cible
        
    Returns:
        (is_valid, message)
    """
    if target in features:
        return False, "⚠️ ERREUR: La variable cible ne peut pas être utilisée comme variable explicative"
    return True, ""

def validate_analysis_requirements(analysis_type, data_info):
    """
    Vérifie si une analyse est possible avec les données disponibles
    
    Args:
        analysis_type: Type d'analyse demandée
        data_info: Dictionnaire contenant les informations sur les données
            - target_type: type de la cible
            - num_features: nombre de features numériques
            - has_date: présence de colonne date
            - has_text: présence de colonne texte
            
    Returns:
        (is_possible, message)
    """
    requirements = {
        'descriptive_stats': {
            'condition': lambda info: info.get('num_features', 0) > 0,
            'message': "Nécessite au moins une variable numérique"
        },
        'correlation': {
            'condition': lambda info: info.get('num_features', 0) >= 2,
            'message': "Nécessite au moins 2 variables numériques"
        },
        'distribution': {
            'condition': lambda info: info.get('num_features', 0) > 0,
            'message': "Nécessite au moins une variable numérique"
        },
        'anomaly_detection': {
            'condition': lambda info: info.get('num_features', 0) > 0,
            'message': "Nécessite au moins une variable numérique"
        },
        'categorical_analysis': {
            'condition': lambda info: info.get('cat_features', 0) > 0,
            'message': "Nécessite au moins une variable catégorielle"
        },
        'regression': {
            'condition': lambda info: info.get('target_type') == 'numeric',
            'message': "Nécessite une cible numérique"
        },
        'classification': {
            'condition': lambda info: info.get('target_type') in ['categorical', 'binary'],
            'message': "Nécessite une cible catégorielle"
        },
        'time_series': {
            'condition': lambda info: info.get('has_date', False),
            'message': "Nécessite une colonne date"
        },
        'text_analysis': {
            'condition': lambda info: info.get('has_text', False),
            'message': "Nécessite une colonne texte"
        },
        'clustering': {
            'condition': lambda info: info.get('num_features', 0) >= 2,
            'message': "Nécessite au moins 2 variables numériques"
        }
    }
    
    if analysis_type not in requirements:
        return True, ""
    
    req = requirements[analysis_type]
    if req['condition'](data_info):
        return True, ""
    else:
        return False, req['message']

def detect_problem_type(target_series):
    """
    Détecte automatiquement le type de problème basé sur la cible
    
    Args:
        target_series: pandas Series de la variable cible
        
    Returns:
        (problem_type, description)
    """
    import pandas as pd
    import numpy as np
    
    # Vérifier si c'est une date
    if pd.api.types.is_datetime64_any_dtype(target_series):
        return 'time_series', 'Séries temporelles'
    
    # Vérifier si c'est numérique
    if pd.api.types.is_numeric_dtype(target_series):
        # Vérifier si c'est discret (peu de valeurs uniques)
        unique_values = target_series.nunique()
        total_values = len(target_series)
        
        if unique_values <= 10 and unique_values < total_values * 0.05:
            # C'est probablement catégoriel
            if unique_values == 2:
                return 'binary_classification', 'Classification binaire'
            else:
                return 'multiclass_classification', 'Classification multiclasse'
        else:
            return 'regression', 'Régression'
    
    # C'est catégoriel
    unique_values = target_series.nunique()
    if unique_values == 2:
        return 'binary_classification', 'Classification binaire'
    else:
        return 'multiclass_classification', 'Classification multiclasse'

def validate_train_test_split(train_size):
    """
    Valide la taille du split train/test
    
    Args:
        train_size: Proportion pour l'ensemble d'entraînement (0-1)
        
    Returns:
        (is_valid, message)
    """
    if not 0.5 <= train_size <= 0.95:
        return False, "La proportion d'entraînement doit être entre 50% et 95%"
    return True, ""

def get_recommended_metrics(problem_type):
    """
    Retourne les métriques recommandées selon le type de problème
    
    Args:
        problem_type: Type de problème détecté
        
    Returns:
        Liste des métriques appropriées
    """
    metrics = {
        'regression': ['R²', 'RMSE', 'MAE', 'MAPE'],
        'binary_classification': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'multiclass_classification': ['Accuracy', 'F1-Score (macro)', 'F1-Score (weighted)'],
        'time_series': ['RMSE', 'MAE', 'MAPE']
    }
    return metrics.get(problem_type, [])
