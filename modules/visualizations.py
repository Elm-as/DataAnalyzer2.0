"""
Visualization module for DataAnalyzer 2.0
Professional charts with matplotlib, seaborn, and plotly
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Configuration du style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

def plot_distribution(data: pd.Series, title: str = "", bins: int = 30, show_kde: bool = True):
    """
    Crée un histogramme avec KDE
    
    Args:
        data: Série de données
        title: Titre du graphique
        bins: Nombre de bins pour l'histogramme
        show_kde: Afficher la courbe KDE
        
    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogramme
    ax.hist(data.dropna(), bins=bins, alpha=0.7, edgecolor='black', density=True, label='Distribution')
    
    # KDE si demandé
    if show_kde and len(data.dropna()) > 1:
        data.dropna().plot.kde(ax=ax, linewidth=2, label='KDE')
    
    ax.set_xlabel(data.name or 'Valeur')
    ax.set_ylabel('Densité')
    ax.set_title(title or f'Distribution de {data.name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_correlation_matrix(corr_matrix: pd.DataFrame, method: str = 'pearson', threshold: float = 0.0):
    """
    Crée une heatmap de corrélation
    
    Args:
        corr_matrix: Matrice de corrélation
        method: Méthode de corrélation
        threshold: Seuil minimum pour afficher
        
    Returns:
        Figure matplotlib
    """
    # Filtrer par seuil
    if threshold > 0:
        mask = np.abs(corr_matrix) < threshold
        corr_matrix = corr_matrix.copy()
        corr_matrix[mask] = 0
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Masque pour la diagonale supérieure
    mask_upper = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, 
                mask=mask_upper,
                annot=True, 
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                vmin=-1, 
                vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=ax)
    
    ax.set_title(f'Matrice de Corrélation ({method.capitalize()})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_categorical_distribution(data: pd.Series, top_n: int = 10, title: str = ""):
    """
    Crée un diagramme en barres pour les variables catégorielles
    
    Args:
        data: Série de données catégorielles
        top_n: Nombre de catégories à afficher
        title: Titre du graphique
        
    Returns:
        Figure matplotlib
    """
    value_counts = data.value_counts().head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("husl", len(value_counts))
    value_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
    
    ax.set_xlabel(data.name or 'Catégorie')
    ax.set_ylabel('Fréquence')
    ax.set_title(title or f'Distribution de {data.name}')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les pourcentages
    total = value_counts.sum()
    for i, (idx, val) in enumerate(value_counts.items()):
        percentage = (val / total) * 100
        ax.text(i, val, f'{percentage:.1f}%', ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_boxplot(data: pd.DataFrame, columns: List[str], title: str = ""):
    """
    Crée des boxplots pour détecter les outliers
    
    Args:
        data: DataFrame
        columns: Liste des colonnes à afficher
        title: Titre du graphique
        
    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data_to_plot = data[columns].dropna()
    
    bp = ax.boxplot([data_to_plot[col] for col in columns],
                     labels=columns,
                     patch_artist=True,
                     notch=True,
                     showmeans=True)
    
    # Colorier les boxplots
    colors = sns.color_palette("Set2", len(columns))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Valeur')
    ax.set_title(title or 'Boxplots - Détection d\'outliers')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_scatter(x: pd.Series, y: pd.Series, hue: Optional[pd.Series] = None, 
                title: str = "", alpha: float = 0.6):
    """
    Crée un nuage de points
    
    Args:
        x: Série pour l'axe X
        y: Série pour l'axe Y
        hue: Série optionnelle pour la coloration
        title: Titre du graphique
        alpha: Transparence des points
        
    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if hue is not None:
        # Scatter avec coloration
        for category in hue.unique():
            mask = hue == category
            ax.scatter(x[mask], y[mask], label=str(category), alpha=alpha, s=50)
        ax.legend()
    else:
        ax.scatter(x, y, alpha=alpha, s=50, color='steelblue')
    
    ax.set_xlabel(x.name or 'X')
    ax.set_ylabel(y.name or 'Y')
    ax.set_title(title or f'{y.name} vs {x.name}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_feature_importance(importances: Dict[str, float], top_n: int = 15, title: str = ""):
    """
    Affiche l'importance des features
    
    Args:
        importances: Dictionnaire {feature: importance}
        top_n: Nombre de features à afficher
        title: Titre du graphique
        
    Returns:
        Figure matplotlib
    """
    # Trier par importance
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, values = zip(*sorted_features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    ax.barh(range(len(features)), values, color=colors, edgecolor='black')
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance')
    ax.set_title(title or 'Importance des Features')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str = ""):
    """
    Affiche une matrice de confusion
    
    Args:
        cm: Matrice de confusion
        labels: Labels des classes
        title: Titre du graphique
        
    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                ax=ax, cbar_kws={"shrink": 0.8})
    
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Vérité')
    ax.set_title(title or 'Matrice de Confusion')
    
    plt.tight_layout()
    return fig

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, title: str = ""):
    """
    Affiche la courbe ROC
    
    Args:
        fpr: False Positive Rate
        tpr: True Positive Rate
        auc: Area Under Curve
        title: Titre du graphique
        
    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title or 'Courbe ROC')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str = ""):
    """
    Affiche les résidus pour la régression
    
    Args:
        y_true: Valeurs réelles
        y_pred: Valeurs prédites
        title: Titre du graphique
        
    Returns:
        Figure matplotlib
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, color='steelblue')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Valeurs prédites')
    axes[0].set_ylabel('Résidus')
    axes[0].set_title('Résidus vs Prédictions')
    axes[0].grid(True, alpha=0.3)
    
    # Distribution des résidus
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Résidus')
    axes[1].set_ylabel('Fréquence')
    axes[1].set_title('Distribution des Résidus')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title or 'Analyse des Résidus', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_learning_curve(train_scores: List[float], val_scores: List[float], 
                        metric_name: str = "Score", title: str = ""):
    """
    Affiche la courbe d'apprentissage
    
    Args:
        train_scores: Scores sur l'ensemble d'entraînement
        val_scores: Scores sur l'ensemble de validation
        metric_name: Nom de la métrique
        title: Titre du graphique
        
    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_scores) + 1)
    
    ax.plot(epochs, train_scores, 'o-', linewidth=2, label='Train', color='blue')
    ax.plot(epochs, val_scores, 's-', linewidth=2, label='Validation', color='orange')
    
    ax.set_xlabel('Époque')
    ax.set_ylabel(metric_name)
    ax.set_title(title or 'Courbe d\'Apprentissage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
