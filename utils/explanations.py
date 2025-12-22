"""
Explanations and pedagogical content for DataAnalyzer 2.0
"""

def get_method_explanation(method_name):
    """
    Retourne l'explication d'une méthode d'analyse
    """
    explanations = {
        'descriptive_stats': {
            'method': """
            Les statistiques descriptives résument les caractéristiques principales des données numériques.
            Elles incluent la moyenne, la médiane, l'écart-type, les quartiles, etc.
            """,
            'interpretation': """
            - **Moyenne**: Valeur centrale des données
            - **Médiane**: Valeur qui divise les données en deux parties égales
            - **Écart-type**: Mesure de la dispersion des données
            - **Min/Max**: Valeurs extrêmes
            - **Quartiles**: Q1 (25%), Q2 (50%), Q3 (75%)
            """,
            'warnings': [
                "La moyenne est sensible aux valeurs extrêmes (outliers)",
                "Préférer la médiane pour des distributions asymétriques"
            ]
        },
        'correlation': {
            'method': """
            La corrélation mesure la relation linéaire entre deux variables numériques.
            - **Pearson**: Pour relations linéaires (données normales)
            - **Spearman**: Pour relations monotones (robuste aux outliers)
            """,
            'interpretation': """
            - **Corrélation > 0.7**: Forte corrélation positive
            - **Corrélation < -0.7**: Forte corrélation négative
            - **Corrélation proche de 0**: Pas de corrélation linéaire
            - **Attention**: Corrélation ≠ Causalité
            """,
            'warnings': [
                "Une corrélation forte ne signifie pas causalité",
                "Des variables peuvent être liées de manière non-linéaire"
            ]
        },
        'distribution': {
            'method': """
            L'analyse de distribution montre comment les valeurs sont réparties.
            - **Histogramme**: Fréquence des valeurs par intervalle
            - **KDE (Kernel Density Estimation)**: Estimation lisse de la densité
            """,
            'interpretation': """
            - **Distribution normale**: Symétrique en forme de cloche
            - **Distribution asymétrique**: Étalée vers la gauche ou la droite
            - **Multimodale**: Plusieurs pics (plusieurs groupes)
            """,
            'warnings': [
                "Vérifier la présence d'outliers",
                "Considérer une transformation si très asymétrique"
            ]
        },
        'anomaly_detection': {
            'method': """
            Détection des valeurs aberrantes (outliers) par la méthode IQR.
            Une valeur est considérée aberrante si:
            - Inférieure à Q1 - 1.5 × IQR
            - Supérieure à Q3 + 1.5 × IQR
            """,
            'interpretation': """
            Les outliers peuvent être:
            - **Erreurs de mesure**: À corriger ou supprimer
            - **Valeurs extrêmes réelles**: À conserver mais analyser
            - **Points intéressants**: Cas particuliers à étudier
            """,
            'warnings': [
                "Ne pas supprimer automatiquement les outliers",
                "Comprendre leur origine avant de les traiter"
            ]
        },
        'regression': {
            'method': """
            La régression prédit une valeur numérique continue.
            Algorithmes disponibles:
            - **Régression Linéaire**: Relation linéaire simple
            - **Random Forest**: Capture les non-linéarités
            - **XGBoost**: Performances élevées, robuste
            """,
            'interpretation': """
            Métriques:
            - **R²**: Pourcentage de variance expliquée (0-1, plus élevé = mieux)
            - **RMSE**: Erreur moyenne en unités de la cible
            - **MAE**: Erreur absolue moyenne (robuste aux outliers)
            """,
            'warnings': [
                "Attention au surapprentissage (overfitting)",
                "Valider sur des données non vues"
            ]
        },
        'classification': {
            'method': """
            La classification prédit une catégorie.
            Algorithmes disponibles:
            - **Régression Logistique**: Modèle simple et interprétable
            - **Random Forest**: Robuste et performant
            - **XGBoost**: Souvent le meilleur en compétition
            """,
            'interpretation': """
            Métriques:
            - **Accuracy**: Taux de bonnes prédictions
            - **Precision**: Proportion de vrais positifs parmi les positifs prédits
            - **Recall**: Proportion de vrais positifs détectés
            - **F1-Score**: Moyenne harmonique de Precision et Recall
            """,
            'warnings': [
                "Attention aux classes déséquilibrées",
                "L'Accuracy peut être trompeuse"
            ]
        },
        'clustering': {
            'method': """
            Le clustering regroupe les données similaires sans supervision.
            - **K-Means**: Rapide, nombre de clusters à définir
            - **DBSCAN**: Trouve les clusters de forme arbitraire
            """,
            'interpretation': """
            Le nombre optimal de clusters peut être déterminé par:
            - **Méthode du coude**: Inertie vs nombre de clusters
            - **Silhouette Score**: Qualité de séparation (-1 à 1)
            """,
            'warnings': [
                "Standardiser les données avant clustering",
                "Le choix du nombre de clusters est crucial"
            ]
        }
    }
    return explanations.get(method_name, {
        'method': 'Méthode non documentée',
        'interpretation': '',
        'warnings': []
    })

def get_titanic_example(analysis_type):
    """
    Retourne un exemple spécifique avec le dataset Titanic
    """
    examples = {
        'target_selection': """
        **Exemple avec Titanic:**
        - Variable cible: **Survived** (0 = Non survécu, 1 = Survécu)
        - Type détecté: Classification binaire
        - Variables explicatives: Age, Sex, Pclass, Fare, etc.
        - ⚠️ Survived est automatiquement exclue des features
        """,
        'correlation': """
        **Exemple avec Titanic:**
        - Forte corrélation négative entre Pclass et Fare (-0.55)
          → Les passagers de 1ère classe payent plus cher
        - Corrélation positive entre Fare et Survived (0.26)
          → Le prix du billet est lié à la survie
        """,
        'classification': """
        **Exemple avec Titanic:**
        - Meilleur modèle: Random Forest (Accuracy ~82%)
        - Features importantes:
          1. Sex (le plus important)
          2. Pclass
          3. Age
          4. Fare
        - Interprétation: Les femmes de 1ère classe ont plus de chances de survie
        """,
        'anomaly_detection': """
        **Exemple avec Titanic:**
        - Outliers détectés dans Fare (quelques billets très chers)
        - Outliers dans Age (personnes âgées)
        - Ces valeurs sont réelles, pas des erreurs
        """
    }
    return examples.get(analysis_type, "")

def get_tips(context):
    """
    Retourne des conseils pratiques selon le contexte
    """
    tips = {
        'large_dataset': """
        📊 **Dataset volumineux détecté (> 10,000 lignes)**
        
        Recommandations:
        - Utiliser l'échantillonnage pour l'exploration
        - Considérer l'échantillonnage stratifié pour conserver les proportions
        - Les calculs peuvent prendre plus de temps
        """,
        'missing_values': """
        ⚠️ **Valeurs manquantes détectées**
        
        Options de traitement:
        - Suppression des lignes (si < 5% de données manquantes)
        - Imputation par la moyenne/médiane (variables numériques)
        - Imputation par le mode (variables catégorielles)
        - Imputation par modèle (plus sophistiqué)
        """,
        'imbalanced_classes': """
        ⚠️ **Classes déséquilibrées détectées**
        
        Solutions:
        - Utiliser des métriques adaptées (F1-Score, ROC-AUC)
        - Techniques de rééquilibrage (SMOTE, under/over-sampling)
        - Ajuster les poids des classes dans le modèle
        """,
        'high_cardinality': """
        ⚠️ **Variable catégorielle avec beaucoup de valeurs uniques**
        
        Recommandations:
        - Regrouper les catégories rares (< 1% des données)
        - Utiliser des techniques d'encodage avancées
        - Considérer comme variable texte si pertinent
        """
    }
    return tips.get(context, "")
