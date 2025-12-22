"""
Machine Learning models module for DataAnalyzer 2.0
RÈGLE STRICTE: La cible ne doit JAMAIS être dans les features
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple
import time
import warnings
warnings.filterwarnings('ignore')

def prepare_features_and_target(df: pd.DataFrame, target: str, features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    RÈGLE 1: Prépare X et y en s'assurant que target n'est PAS dans features
    
    Args:
        df: DataFrame complet
        target: Nom de la variable cible
        features: Liste des features (NE DOIT PAS contenir target)
        
    Returns:
        (X, y) avec X ne contenant JAMAIS la cible
    """
    # VALIDATION CRITIQUE
    if target in features:
        raise ValueError(f"❌ ERREUR CRITIQUE: La variable cible '{target}' ne peut pas être dans les features!")
    
    # S'assurer que toutes les features existent
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Features manquantes: {missing_features}")
    
    # Extraire X et y
    X = df[features].copy()
    y = df[target].copy()
    
    return X, y

def train_regression_model(df: pd.DataFrame, target: str, features: List[str], 
                          params: Dict = {}) -> Dict:
    """
    Entraîne des modèles de régression
    
    Args:
        df: DataFrame
        target: Nom de la variable cible (numérique)
        features: Liste des features (SANS la cible)
        params: Paramètres (test_size, random_state, models, scale)
        
    Returns:
        Dictionnaire des résultats
    """
    start_time = time.time()
    
    try:
        # RÈGLE 1: Vérifier que target n'est pas dans features
        if target in features:
            return {
                'success': False,
                'error': "❌ La variable cible ne peut pas être dans les features!",
                'execution_time': time.time() - start_time
            }
        
        # Paramètres
        test_size = params.get('test_size', 0.2)
        random_state = params.get('random_state', 42)
        scale_features = params.get('scale', True)
        selected_models = params.get('models', ['linear', 'random_forest', 'xgboost'])
        
        # Préparer les données
        X, y = prepare_features_and_target(df, target, features)
        
        # Supprimer les lignes avec valeurs manquantes
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            return {
                'success': False,
                'error': "Aucune donnée valide après suppression des valeurs manquantes",
                'execution_time': time.time() - start_time
            }
        
        # Encoder les variables catégorielles
        X_encoded = X.copy()
        encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, random_state=random_state
        )
        
        # Standardisation
        scaler = None
        if scale_features:
            scaler = StandardScaler()
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        # Modèles disponibles
        models_dict = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=random_state, n_jobs=-1, verbose=-1)
        }
        
        # Entraîner les modèles
        results_models = {}
        best_model = None
        best_score = -np.inf
        
        for model_name in selected_models:
            if model_name not in models_dict:
                continue
            
            model = models_dict[model_name]
            
            # Entraînement
            model.fit(X_train, y_train)
            
            # Prédictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Métriques
            train_metrics = {
                'r2': r2_score(y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'mae': mean_absolute_error(y_train, y_pred_train)
            }
            
            test_metrics = {
                'r2': r2_score(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae': mean_absolute_error(y_test, y_pred_test)
            }
            
            # Feature importance
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                for feat, imp in zip(X_train.columns, model.feature_importances_):
                    feature_importance[feat] = float(imp)
            elif hasattr(model, 'coef_'):
                for feat, coef in zip(X_train.columns, model.coef_):
                    feature_importance[feat] = float(abs(coef))
            
            results_models[model_name] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': feature_importance,
                'model': model
            }
            
            # Garder le meilleur modèle (basé sur R² test)
            if test_metrics['r2'] > best_score:
                best_score = test_metrics['r2']
                best_model = model_name
        
        results = {
            'success': True,
            'results': {
                'models': results_models,
                'best_model': best_model,
                'best_score': best_score,
                'target': target,
                'features': features,
                'n_features': len(features),
                'n_train': len(X_train),
                'n_test': len(X_test),
                'scaler': scaler,
                'encoders': encoders
            },
            'visualizations': ['feature_importance', 'residuals'],
            'explanations': {
                'method': f"Régression avec {len(selected_models)} modèles entraînés",
                'interpretation': f"""
                Meilleur modèle: {best_model} (R² = {best_score:.3f})
                - R² mesure la proportion de variance expliquée (0-1, 1 = parfait)
                - RMSE mesure l'erreur moyenne en unités de la cible
                - MAE est robuste aux outliers
                """,
                'warnings': [
                    f"Séparation train/test: {100*(1-test_size):.0f}%/{100*test_size:.0f}%",
                    "Valider sur des données non vues"
                ]
            },
            'python_code': f"""
# Code généré pour reproduire l'analyse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Charger les données
df = pd.read_csv('votre_fichier.csv')

# RÈGLE: La cible n'est JAMAIS dans les features
features = {features}
target = '{target}'
X = df[features]
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state={random_state}
)

# Entraîner le modèle
model = RandomForestRegressor(n_estimators=100, random_state={random_state})
model.fit(X_train, y_train)

# Évaluer
score = model.score(X_test, y_test)
print(f'R² Score: {{{{score:.3f}}}}')
""",
            'execution_time': time.time() - start_time
        }
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'execution_time': time.time() - start_time
        }

def train_classification_model(df: pd.DataFrame, target: str, features: List[str],
                               params: Dict = {}) -> Dict:
    """
    Entraîne des modèles de classification
    
    Args:
        df: DataFrame
        target: Nom de la variable cible (catégorielle)
        features: Liste des features (SANS la cible)
        params: Paramètres
        
    Returns:
        Dictionnaire des résultats
    """
    start_time = time.time()
    
    try:
        # RÈGLE 1: Vérifier que target n'est pas dans features
        if target in features:
            return {
                'success': False,
                'error': "❌ La variable cible ne peut pas être dans les features!",
                'execution_time': time.time() - start_time
            }
        
        # Paramètres
        test_size = params.get('test_size', 0.2)
        random_state = params.get('random_state', 42)
        scale_features = params.get('scale', True)
        selected_models = params.get('models', ['logistic', 'random_forest', 'xgboost'])
        
        # Préparer les données
        X, y = prepare_features_and_target(df, target, features)
        
        # Supprimer les lignes avec valeurs manquantes
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            return {
                'success': False,
                'error': "Aucune donnée valide après suppression des valeurs manquantes",
                'execution_time': time.time() - start_time
            }
        
        # Encoder la cible si nécessaire
        target_encoder = None
        if y.dtype == 'object' or y.dtype.name == 'category':
            target_encoder = LabelEncoder()
            y = pd.Series(target_encoder.fit_transform(y), index=y.index)
        
        # Encoder les features catégorielles
        X_encoded = X.copy()
        encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Standardisation
        scaler = None
        if scale_features:
            scaler = StandardScaler()
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        # Modèles disponibles
        n_classes = len(np.unique(y))
        models_dict = {
            'logistic': LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=-1),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=random_state, n_jobs=-1, use_label_encoder=False, eval_metric='logloss'),
            'lightgbm': lgb.LGBMClassifier(n_estimators=100, random_state=random_state, n_jobs=-1, verbose=-1)
        }
        
        # Entraîner les modèles
        results_models = {}
        best_model = None
        best_score = -np.inf
        
        for model_name in selected_models:
            if model_name not in models_dict:
                continue
            
            model = models_dict[model_name]
            
            # Entraînement
            model.fit(X_train, y_train)
            
            # Prédictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Prédictions probabilistes (si disponibles)
            y_pred_proba_test = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba_test = model.predict_proba(X_test)
            
            # Métriques
            train_metrics = {
                'accuracy': accuracy_score(y_train, y_pred_train),
                'precision': precision_score(y_train, y_pred_train, average='weighted', zero_division=0),
                'recall': recall_score(y_train, y_pred_train, average='weighted', zero_division=0),
                'f1': f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
            }
            
            test_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_test),
                'precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            }
            
            # ROC-AUC pour classification binaire
            if n_classes == 2 and y_pred_proba_test is not None:
                test_metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba_test[:, 1])
            
            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred_test)
            
            # Feature importance
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                for feat, imp in zip(X_train.columns, model.feature_importances_):
                    feature_importance[feat] = float(imp)
            elif hasattr(model, 'coef_'):
                for feat, coef in zip(X_train.columns, np.abs(model.coef_).mean(axis=0)):
                    feature_importance[feat] = float(coef)
            
            results_models[model_name] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'confusion_matrix': cm.tolist(),
                'feature_importance': feature_importance,
                'model': model
            }
            
            # Garder le meilleur modèle (basé sur F1 test)
            if test_metrics['f1'] > best_score:
                best_score = test_metrics['f1']
                best_model = model_name
        
        # Classes
        class_labels = target_encoder.classes_.tolist() if target_encoder else sorted(y.unique())
        
        results = {
            'success': True,
            'results': {
                'models': results_models,
                'best_model': best_model,
                'best_score': best_score,
                'target': target,
                'features': features,
                'n_features': len(features),
                'n_train': len(X_train),
                'n_test': len(X_test),
                'n_classes': n_classes,
                'class_labels': class_labels,
                'scaler': scaler,
                'encoders': encoders,
                'target_encoder': target_encoder
            },
            'visualizations': ['feature_importance', 'confusion_matrix'],
            'explanations': {
                'method': f"Classification avec {len(selected_models)} modèles entraînés",
                'interpretation': f"""
                Meilleur modèle: {best_model} (F1-Score = {best_score:.3f})
                - Accuracy: Taux de bonnes prédictions
                - Precision: Proportion de vrais positifs
                - Recall: Proportion de positifs détectés
                - F1: Moyenne harmonique Precision/Recall
                """,
                'warnings': [
                    f"Classes: {class_labels}",
                    "Attention aux classes déséquilibrées"
                ]
            },
            'python_code': f"""
# Code généré pour reproduire l'analyse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Charger les données
df = pd.read_csv('votre_fichier.csv')

# RÈGLE: La cible n'est JAMAIS dans les features
features = {features}
target = '{target}'
X = df[features]
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state={random_state}, stratify=y
)

# Entraîner le modèle
model = RandomForestClassifier(n_estimators=100, random_state={random_state})
model.fit(X_train, y_train)

# Évaluer
score = model.score(X_test, y_test)
print(f'Accuracy: {{{{score:.3f}}}}')
""",
            'execution_time': time.time() - start_time
        }
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'execution_time': time.time() - start_time
        }

def clustering_analysis(df: pd.DataFrame, features: List[str], params: Dict = {}) -> Dict:
    """
    Analyse de clustering (non supervisé, pas de cible)
    
    Args:
        df: DataFrame
        features: Liste des features numériques
        params: Paramètres (n_clusters, method)
        
    Returns:
        Dictionnaire des résultats
    """
    start_time = time.time()
    
    try:
        n_clusters = params.get('n_clusters', 3)
        method = params.get('method', 'kmeans')
        scale_features = params.get('scale', True)
        
        # Sélectionner les features
        X = df[features].copy()
        
        # Supprimer les lignes avec valeurs manquantes
        X = X.dropna()
        
        if len(X) == 0:
            return {
                'success': False,
                'error': "Aucune donnée valide",
                'execution_time': time.time() - start_time
            }
        
        # Standardisation
        if scale_features:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values
        
        # Clustering
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
            centers = model.cluster_centers_
        elif method == 'dbscan':
            eps = params.get('eps', 0.5)
            min_samples = params.get('min_samples', 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_scaled)
            centers = None
        else:
            return {
                'success': False,
                'error': f"Méthode inconnue: {method}",
                'execution_time': time.time() - start_time
            }
        
        # Résultats
        unique_labels = np.unique(labels)
        cluster_sizes = {int(label): int(np.sum(labels == label)) for label in unique_labels}
        
        results = {
            'success': True,
            'results': {
                'labels': labels.tolist(),
                'n_clusters': len(unique_labels),
                'cluster_sizes': cluster_sizes,
                'method': method,
                'features': features
            },
            'visualizations': ['scatter_clusters'],
            'explanations': {
                'method': f"Clustering {method} avec {n_clusters} clusters",
                'interpretation': """
                Le clustering regroupe les observations similaires.
                - K-Means: Clusters sphériques
                - DBSCAN: Clusters de forme arbitraire
                """,
                'warnings': [
                    'Standardiser les données avant clustering',
                    'Le choix du nombre de clusters est crucial'
                ]
            },
            'execution_time': time.time() - start_time
        }
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'execution_time': time.time() - start_time
        }
