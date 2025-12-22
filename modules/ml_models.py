"""
Machine Learning models module for DataAnalyzer 2.0
RÈGLE STRICTE: La cible ne doit JAMAIS être dans les features
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
from sklearn.metrics import silhouette_score
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb
from typing import Any, Dict, List, Optional, Tuple
import time
import warnings
import statsmodels.api as sm
warnings.filterwarnings('ignore')


class SafeLabelEncoder:
    """Encodeur simple (type LabelEncoder) tolérant aux catégories inconnues.

    - Fit sur les catégories vues.
    - Transform mappe les inconnues vers unknown_value (par défaut -1).
    - Pickle-friendly (utilisé dans ModelBundle).
    """

    def __init__(self, unknown_value: int = -1, missing_token: str = '__MISSING__'):
        self.unknown_value = int(unknown_value)
        self.missing_token = str(missing_token)
        self._map: dict[str, int] = {}
        self.classes_: list[str] = []

    def fit(self, values) -> 'SafeLabelEncoder':
        s = pd.Series(values).astype('string').fillna(self.missing_token)
        cats = pd.unique(s)
        # Tri stable pour reproductibilité
        cats = [str(x) for x in cats.tolist()]
        cats_sorted = sorted(cats)
        self.classes_ = cats_sorted
        self._map = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, values) -> np.ndarray:
        if not self._map:
            raise ValueError('SafeLabelEncoder non entraîné (fit manquant).')
        s = pd.Series(values).astype('string').fillna(self.missing_token)
        return s.map(lambda x: self._map.get(str(x), self.unknown_value)).astype('int64').to_numpy()

    def fit_transform(self, values) -> np.ndarray:
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values) -> np.ndarray:
        # Les unknown_value ne peuvent pas être décodées -> '__UNKNOWN__'
        inv = {i: c for c, i in self._map.items()}
        out = []
        for v in values:
            try:
                iv = int(v)
            except Exception:
                out.append('__UNKNOWN__')
                continue
            out.append(inv.get(iv, '__UNKNOWN__'))
        return np.array(out, dtype=object)


def _impute_and_encode(
    X: pd.DataFrame,
    encoders: Optional[Dict[str, Any]] = None,
    impute_values: Optional[Dict[str, float]] = None,
    missing_token: str = '__MISSING__',
    fit: bool = False,
) -> tuple[pd.DataFrame, Dict[str, Any], Dict[str, float]]:
    """Impute + encode, avec option fit sur X.

    - Colonnes object/category => SafeLabelEncoder
    - Colonnes numériques => pd.to_numeric + imputation médiane

    Returns: (X_encoded, encoders, impute_values)
    """
    X2 = X.copy()
    enc = {} if encoders is None else dict(encoders)
    imps: Dict[str, float] = {} if impute_values is None else dict(impute_values)

    for col in X2.columns:
        is_cat = (X2[col].dtype == 'object') or (X2[col].dtype.name == 'category')
        if is_cat:
            if fit or (col not in enc):
                le = SafeLabelEncoder(unknown_value=-1, missing_token=missing_token)
                X2[col] = le.fit_transform(X2[col])
                enc[col] = le
            else:
                le = enc[col]
                X2[col] = le.transform(X2[col])
        else:
            # Numérique (ou convertible)
            s = pd.to_numeric(X2[col], errors='coerce')
            if fit or (col not in imps):
                med = float(s.median()) if s.notna().any() else 0.0
                imps[col] = med
            s = s.fillna(imps.get(col, 0.0))
            X2[col] = s

    return X2, enc, imps

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
        raise ValueError(f"ERREUR: La variable cible '{target}' ne peut pas être dans les features")
    
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
                'error': "La variable cible ne peut pas être dans les features",
                'execution_time': time.time() - start_time
            }
        
        # Paramètres
        test_size = params.get('test_size', 0.2)
        random_state = params.get('random_state', 42)
        scale_features = params.get('scale', True)
        selected_models = params.get('models', ['linear', 'random_forest', 'xgboost'])
        
        # Préparer les données
        X, y = prepare_features_and_target(df, target, features)
        
        # Split train/test (AVANT encodage) pour éviter fuite + gérer catégories inconnues en test
        mask_y = ~y.isna()
        X = X.loc[mask_y]
        y = y.loc[mask_y]

        if len(X) < 10:
            return {
                'success': False,
                'error': "Jeu de données insuffisant pour entraîner un modèle (>=10 lignes recommandées).",
                'execution_time': time.time() - start_time
            }

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Imputer + encoder (fit sur train)
        X_train, encoders, impute_values = _impute_and_encode(X_train_raw, fit=True)
        X_test, _, _ = _impute_and_encode(X_test_raw, encoders=encoders, impute_values=impute_values, fit=False)
        
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

            # Hétéroscédasticité (Breusch-Pagan) sur test (optionnel)
            try:
                resid = (pd.Series(y_test, index=X_test.index) - pd.Series(y_pred_test, index=X_test.index)).astype(float)
                exog = sm.add_constant(pd.DataFrame(X_test, index=X_test.index))
                lm_stat, lm_pvalue, f_stat, f_pvalue = sm.stats.diagnostic.het_breuschpagan(resid, exog)
                test_metrics['breusch_pagan'] = {
                    'lm_stat': float(lm_stat),
                    'lm_pvalue': float(lm_pvalue),
                    'f_stat': float(f_stat),
                    'f_pvalue': float(f_pvalue),
                }
            except Exception:
                pass
            
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
                'encoders': encoders,
                'impute_values': impute_values,
                'missing_token': '__MISSING__',
                'diagnostics': {},
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
        
        _add_advanced_regression_diagnostics(results, X_train, y_train, X_test, y_test)
        return results
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'execution_time': time.time() - start_time
        }


def _add_advanced_regression_diagnostics(results: Dict, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Ajoute des diagnostics avancés au dict results['results']['diagnostics'] (in-place)."""
    try:
        r = results.get('results') or {}
        best_name = r.get('best_model')
        models = r.get('models') or {}
        if not best_name or best_name not in models:
            return
        model = models[best_name].get('model')
        if model is None:
            return

        diag: Dict[str, Any] = dict(r.get('diagnostics') or {})

        # Cross-validation
        try:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
            diag['cv_r2'] = {'mean': float(np.mean(scores)), 'std': float(np.std(scores))}
        except Exception:
            pass

        # Learning curve (limiter sur gros dataset)
        try:
            if len(X_train) <= 3000:
                train_sizes = np.linspace(0.1, 1.0, 5)
                ts, train_scores, val_scores = learning_curve(
                    model,
                    X_train,
                    y_train,
                    cv=5,
                    scoring='r2',
                    train_sizes=train_sizes,
                    n_jobs=-1,
                )
                diag['learning_curve_r2'] = {
                    'train_sizes': ts.tolist(),
                    'train_mean': np.mean(train_scores, axis=1).tolist(),
                    'train_std': np.std(train_scores, axis=1).tolist(),
                    'val_mean': np.mean(val_scores, axis=1).tolist(),
                    'val_std': np.std(val_scores, axis=1).tolist(),
                }
        except Exception:
            pass

        # Permutation importance (test)
        try:
            perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
            imp = {str(f): float(v) for f, v in zip(X_test.columns, perm.importances_mean)}
            diag['permutation_importance'] = imp
        except Exception:
            pass

        r['diagnostics'] = diag
        results['results'] = r
    except Exception:
        return

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
                'error': "La variable cible ne peut pas être dans les features",
                'execution_time': time.time() - start_time
            }
        
        # Paramètres
        test_size = params.get('test_size', 0.2)
        random_state = params.get('random_state', 42)
        scale_features = params.get('scale', True)
        selected_models = params.get('models', ['logistic', 'random_forest', 'xgboost'])
        
        # Préparer les données
        X, y = prepare_features_and_target(df, target, features)
        
        # Retirer lignes sans cible
        mask_y = ~y.isna()
        X = X.loc[mask_y]
        y = y.loc[mask_y]

        if len(X) < 10:
            return {
                'success': False,
                'error': "Jeu de données insuffisant pour entraîner un modèle (>=10 lignes recommandées).",
                'execution_time': time.time() - start_time
            }

        # Split train/test (AVANT encodage)
        # Stratify peut échouer si classe rare -> fallback
        try:
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except Exception:
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        # Encoder la cible sur train
        target_encoder = None
        if (y_train_raw.dtype == 'object') or (y_train_raw.dtype.name == 'category'):
            target_encoder = LabelEncoder()
            y_train = pd.Series(target_encoder.fit_transform(y_train_raw.astype(str)), index=y_train_raw.index)
            # test: valeurs inconnues -> -1
            try:
                y_test = pd.Series(target_encoder.transform(y_test_raw.astype(str)), index=y_test_raw.index)
            except Exception:
                # si des classes n'apparaissent qu'en test (rare) : marquer -1
                mapping = {str(c): i for i, c in enumerate(getattr(target_encoder, 'classes_', []))}
                y_test = pd.Series([mapping.get(str(v), -1) for v in y_test_raw.astype(str)], index=y_test_raw.index)
        else:
            y_train = y_train_raw
            y_test = y_test_raw

        # Imputer + encoder features (fit sur train)
        X_train, encoders, impute_values = _impute_and_encode(X_train_raw, fit=True)
        X_test, _, _ = _impute_and_encode(X_test_raw, encoders=encoders, impute_values=impute_values, fit=False)
        
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
        n_classes = len(np.unique(y_train))
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

                # Calibration (binaire)
                try:
                    test_metrics['brier'] = float(brier_score_loss(y_test, y_pred_proba_test[:, 1]))
                    frac_pos, mean_pred = calibration_curve(y_test, y_pred_proba_test[:, 1], n_bins=10, strategy='uniform')
                    test_metrics['calibration_curve'] = {
                        'mean_pred': [float(x) for x in mean_pred],
                        'frac_pos': [float(x) for x in frac_pos],
                    }
                except Exception:
                    pass
            
            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred_test)
            try:
                cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
                cm_norm_list = cm_norm.tolist()
            except Exception:
                cm_norm_list = None
            
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
                'confusion_matrix_normalized': cm_norm_list,
                'feature_importance': feature_importance,
                'model': model
            }
            
            # Garder le meilleur modèle (basé sur F1 test)
            if test_metrics['f1'] > best_score:
                best_score = test_metrics['f1']
                best_model = model_name
        
        # Classes
        class_labels = target_encoder.classes_.tolist() if target_encoder else sorted(pd.Series(y_train).unique())
        
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
                'target_encoder': target_encoder,
                'impute_values': impute_values,
                'missing_token': '__MISSING__',
                'diagnostics': {},
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
        
        # Diagnostics avancés (best model)
        _add_advanced_classification_diagnostics(results, X_train, y_train, X_test, y_test)
        return results
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'execution_time': time.time() - start_time
        }


def _add_advanced_classification_diagnostics(results: Dict, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Ajoute des diagnostics avancés au dict results['results']['diagnostics'] (in-place)."""
    try:
        r = results.get('results') or {}
        best_name = r.get('best_model')
        models = r.get('models') or {}
        if not best_name or best_name not in models:
            return
        model = models[best_name].get('model')
        if model is None:
            return

        diag: Dict[str, Any] = dict(r.get('diagnostics') or {})

        # Top erreurs (test)
        try:
            y_pred = pd.Series(model.predict(X_test), index=X_test.index)
            y_true = pd.Series(y_test, index=X_test.index)
            wrong_mask = (y_pred != y_true)
            wrong_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})[wrong_mask].head(25)
            examples = []
            for idx, row in wrong_df.iterrows():
                examples.append({
                    'index': int(idx) if str(idx).isdigit() else str(idx),
                    'y_true': int(row['y_true']) if pd.notna(row['y_true']) else None,
                    'y_pred': int(row['y_pred']) if pd.notna(row['y_pred']) else None,
                })
            diag['top_errors'] = {
                'count': int(wrong_mask.sum()),
                'examples': examples,
            }
        except Exception:
            pass

        # Cross-validation (weighted F1)
        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
            diag['cv_f1_weighted'] = {'mean': float(np.mean(scores)), 'std': float(np.std(scores))}
        except Exception:
            pass

        # Learning curve (limiter sur gros dataset)
        try:
            if len(X_train) <= 3000:
                train_sizes = np.linspace(0.1, 1.0, 5)
                ts, train_scores, val_scores = learning_curve(
                    model,
                    X_train,
                    y_train,
                    cv=5,
                    scoring='f1_weighted',
                    train_sizes=train_sizes,
                    n_jobs=-1,
                )
                diag['learning_curve_f1_weighted'] = {
                    'train_sizes': ts.tolist(),
                    'train_mean': np.mean(train_scores, axis=1).tolist(),
                    'train_std': np.std(train_scores, axis=1).tolist(),
                    'val_mean': np.mean(val_scores, axis=1).tolist(),
                    'val_std': np.std(val_scores, axis=1).tolist(),
                }
        except Exception:
            pass

        # Permutation importance (test)
        try:
            perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
            imp = {str(f): float(v) for f, v in zip(X_test.columns, perm.importances_mean)}
            diag['permutation_importance'] = imp
        except Exception:
            pass

        r['diagnostics'] = diag
        results['results'] = r
    except Exception:
        return

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

        # Forcer en numérique + imputer médiane
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            if X[col].notna().any():
                X[col] = X[col].fillna(float(X[col].median()))
            else:
                X[col] = X[col].fillna(0.0)

        if len(X) < 10:
            return {
                'success': False,
                'error': "Clustering: jeu de données insuffisant (>=10 lignes recommandées).",
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
        
        # Diagnostics
        unique_labels = np.unique(labels)
        cluster_sizes = {int(label): int(np.sum(labels == label)) for label in unique_labels}

        # Silhouette (si applicable)
        silhouette = None
        try:
            # pour DBSCAN, ignorer bruit (-1) et exiger >=2 clusters
            lab = np.asarray(labels)
            mask = lab != -1
            unique_eff = np.unique(lab[mask])
            if len(unique_eff) >= 2 and int(mask.sum()) >= 10:
                silhouette = float(silhouette_score(np.asarray(X_scaled)[mask], lab[mask]))
        except Exception:
            silhouette = None

        # Elbow (KMeans uniquement)
        elbow = None
        if method == 'kmeans':
            try:
                max_k = int(params.get('max_k', 10))
                max_k = max(3, min(max_k, 15, len(X) - 1))
                ks = list(range(2, max_k + 1))
                inertias = []
                for k in ks:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    km.fit(X_scaled)
                    inertias.append(float(km.inertia_))
                elbow = {'k': ks, 'inertia': inertias}
            except Exception:
                elbow = None

        # PCA 2D (échantillon pour export/inline)
        pca_payload = None
        try:
            n_max = int(params.get('pca_max_points', 2000))
            if len(X_scaled) > n_max:
                idx = np.linspace(0, len(X_scaled) - 1, n_max).astype(int)
                X_sub = np.asarray(X_scaled)[idx]
                lab_sub = np.asarray(labels)[idx]
            else:
                X_sub = np.asarray(X_scaled)
                lab_sub = np.asarray(labels)
            comps = PCA(n_components=2, random_state=42).fit_transform(X_sub)
            pca_payload = {
                'x': comps[:, 0].astype(float).tolist(),
                'y': comps[:, 1].astype(float).tolist(),
                'labels': lab_sub.astype(int).tolist(),
            }
        except Exception:
            pca_payload = None

        # Profils par cluster (moyennes/medians)
        profiles = None
        try:
            df_prof = X.copy()
            df_prof['_cluster'] = labels
            grp = df_prof.groupby('_cluster')
            means = grp[features].mean(numeric_only=True).to_dict(orient='index')
            meds = grp[features].median(numeric_only=True).to_dict(orient='index')
            profiles = {
                'mean': {str(k): {str(f): float(v) for f, v in d.items()} for k, d in means.items()},
                'median': {str(k): {str(f): float(v) for f, v in d.items()} for k, d in meds.items()},
            }
        except Exception:
            profiles = None
        
        results = {
            'success': True,
            'results': {
                'labels': labels.tolist(),
                'n_clusters': len(unique_labels),
                'cluster_sizes': cluster_sizes,
                'method': method,
                'features': features,
                'silhouette': silhouette,
                'elbow': elbow,
                'pca_2d': pca_payload,
                'cluster_profiles': profiles,
            },
            'visualizations': ['scatter_clusters', 'elbow_curve', 'cluster_sizes'],
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
            'python_code': f"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

df = pd.read_csv('votre_fichier.csv')
features = {features}
X = df[features].apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.median(numeric_only=True)).fillna(0)

X_scaled = StandardScaler().fit_transform(X)

method = '{method}'
if method == 'kmeans':
    model = KMeans(n_clusters={n_clusters}, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
else:
    model = DBSCAN(eps={params.get('eps', 0.5)}, min_samples={params.get('min_samples', 5)})
    labels = model.fit_predict(X_scaled)

mask = labels != -1
if len(set(labels[mask])) >= 2:
    print('Silhouette:', silhouette_score(X_scaled[mask], labels[mask]))
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
