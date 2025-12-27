# DataAnalyzer 2.0 - Interface Assistant Pas à Pas

## Vue d'ensemble

DataAnalyzer 2.0 a été transformé en un assistant intelligent qui guide l'utilisateur à travers 7 étapes pour une analyse de données complète.

## Architecture

### Structure des fichiers

```
modules/dashboard/
├── wizard_views.py         # Logique des 7 étapes du wizard
├── views.py                # Interface classique (backup)
├── forms.py                # Formulaires Django
├── services.py             # Services de session
└── ml_storage.py           # Stockage des modèles

templates/wizard/
├── base.html               # Template de base avec stepper
├── step0_welcome.html      # Page d'accueil
├── step1_import.html       # Import des données
├── step2_preview.html      # Aperçu des données
├── step3_quality.html      # Rapport de qualité
├── step4_configuration.html # Configuration (cible, types, features)
├── step5_analysis_selection.html # Sélection des analyses
├── step6_results.html      # Résultats par onglets
└── step7_simulation.html   # Simulation et prédiction

static/css/
└── wizard.css              # Styles personnalisés pour le wizard
```

## Les 7 Étapes

### Étape 0 : Accueil
- Page d'accueil moderne avec présentation
- Guide des 7 étapes
- Bouton "Commencer l'analyse"
- Fonctionnalités clés mises en avant

### Étape 1 : Import des Données
- Upload de fichiers (CSV, Excel, JSON)
- Datasets d'exemple (Titanic, IRIS)
- Validation du format
- Affichage des informations rapides

**Validation** : Doit avoir des données chargées pour passer à l'étape 2

### Étape 2 : Aperçu des Données
- Affichage des 20 premières lignes
- Vue tabulaire avec types de colonnes
- Métriques rapides (lignes, colonnes, types)
- Identification des colonnes problématiques

**Navigation** : Accès libre, peut revenir pour recharger des données

### Étape 3 : Rapport de Qualité
- **Score de qualité global** (0-100%) avec indicateur visuel
- **Métriques de complétude** :
  - Lignes complètes
  - Valeurs N/A
  - Doublons
- **Détail des colonnes** avec état (OK, Attention, Problématique)
- **Avertissements** automatiques :
  - Lignes dupliquées détectées
  - Colonnes problématiques identifiées
- **Suggestions** personnalisées :
  - Colonnes à nettoyer/supprimer
  - Qualité suffisante ou non pour l'analyse

**Innovation** : Génération automatique d'avertissements et suggestions basés sur les seuils de qualité

### Étape 4 : Configuration des Données
- **Section 1 : Sélection de la variable cible**
  - Liste déroulante de toutes les colonnes
  - Détection automatique du type de problème (régression/classification)
  
- **Section 2 : Vérification et modification des types**
  - Tableau avec type détecté vs type à utiliser
  - Modification manuelle possible (numérique, catégorielle, texte, date, booléen)
  
- **Section 3 : Sélection des features**
  - Exclusion automatique de la cible
  - Barre de recherche pour filtrer les colonnes
  - Boutons "Tout sélectionner" / "Tout désélectionner"
  - Recommandation de désélectionner IDs, noms, cabines, etc.

**Validation** : Doit avoir une cible sélectionnée pour passer à l'étape 5

### Étape 5 : Options d'Analyse
- **Métriques de synthèse** :
  - Lignes de données
  - Colonnes sélectionnées
  - Colonnes numériques
  - Temps estimé total

- **Analyses de Base** :
  - Statistiques descriptives (~2s)
  - Corrélations (~3s)
  - Distributions (~4s)
  - Détection d'anomalies (~3s)
  - Analyse catégorielle (~3s)

- **Analyses Avancées (ML/DL)** :
  - Régression ML (~10s)
  - Classification ML (~12s)
  - Clustering Avancé (~12s)
  - Séries Temporelles (~20s)

**Activation conditionnelle** :
- Analyses basiques : selon la présence de colonnes numériques/catégorielles
- Régression : si cible numérique
- Classification : si cible catégorielle
- Séries temporelles : si colonne date présente
- Clustering : si au moins 2 colonnes numériques

**Interface** :
- Cartes visuelles pour chaque analyse
- État (activé/désactivé) avec icônes
- Durée estimée
- Sélection multiple avec compteur
- Boutons de sélection groupée

**Validation** : Doit avoir au moins une analyse sélectionnée

### Étape 6 : Résultats
- **Onglet Résumé** :
  - Liste des analyses complétées
  - Meilleur modèle (si ML)
  - Score et temps d'exécution
  
- **Onglets individuels** pour chaque analyse :
  - Statistiques descriptives
  - Corrélations
  - Distributions
  - Anomalies
  - Catégorielles
  - Régression/Classification
  - Clustering
  - Séries temporelles

- **Options d'export** :
  - Rapport HTML
  - Rapport PDF
  - Code Python reproductible
  - Bundle complet (ZIP)

**Métriques détaillées** (voir section suivante)

### Étape 7 : Simulation & Prédiction
- **Formulaire dynamique** généré selon les features
- **Types intelligents** : numérique vs texte
- **Exclusion automatique de la cible**
- **Résultat de prédiction** :
  - Valeur prédite
  - Probabilités par classe (classification)
  - Interprétation

**Validation** : Accessible uniquement si un modèle ML a été entraîné

## Gestion de Session

Le wizard utilise la session Django pour stocker :
- `wizard_step` : Étape actuelle (0-7)
- `wizard_completed_steps` : Liste des étapes complétées
- `wizard_selected_analyses` : Analyses sélectionnées
- `wizard_quality_warnings` : Avertissements de qualité
- `wizard_quality_suggestions` : Suggestions de qualité
- `wizard_analysis_results` : Résultats de toutes les analyses

## Navigation et Validation

### Contrôle d'accès
Chaque étape vérifie si l'utilisateur peut y accéder via `_can_access_step()` :
- Étape 0-1 : Toujours accessible
- Étape 2-4 : Nécessite des données chargées
- Étape 5 : Nécessite cible sélectionnée
- Étape 6 : Nécessite analyses sélectionnées
- Étape 7 : Nécessite modèle ML entraîné

### Stepper visuel
- Affiche les 7 étapes avec numéros
- États : actif, complété, futur
- Indicateur de progression visuel
- Responsive pour mobile

### Boutons de navigation
- **Précédent** : Retour à l'étape précédente (sauf étape 1)
- **Suivant** : Avance à l'étape suivante (désactivé si validation échoue)
- **Suivant personnalisé** : Certaines étapes ont des boutons contextuels

## Métriques Détaillées par Analyse

### Statistiques Descriptives
**Pour chaque variable numérique :**
- Count (effectif)
- Mean (moyenne)
- Median (médiane)
- Std (écart-type)
- Min / Max
- Q1 / Q2 / Q3
- IQR (écart interquartile)
- Variance
- Kurtosis
- Skewness
- Missing rate
- Unique values count
- Coefficient of variation

**Pour catégorielles :**
- Effectifs par modalité
- Fréquences %
- Mode
- Cardinalité
- Rare labels

### Corrélations
- Pearson r (corrélation linéaire)
- Spearman ρ (rang)
- P-value associée
- Matrice de corrélation
- Heatmap
- Tests H0 : corrélation = 0
- Intervalle de confiance

### Distributions
- Histogramme
- Densité (KDE)
- Boxplot
- Outliers détectés
- Test de Shapiro-Wilk
- P-value
- QQ-Plot

### Détection d'Anomalies (IQR)
- Limite inférieure = Q1 − 1.5×IQR
- Limite supérieure = Q3 + 1.5×IQR
- Nombre d'outliers
- Pourcentage d'outliers
- Liste des observations extrêmes

### Analyse Catégorielle
- Effectifs / fréquences
- Top-k catégories
- Entropie
- Rare category detection
- Croisements simples
- Moyenne par modalité

### Tests d'Association (Chi-2)
- Khi-deux statistic
- ddl (degrees of freedom)
- P-value
- Effectif observé vs théorique
- Résidus de Pearson
- Cramer's V
- Phi coefficient

### Régression (ML)
- RMSE
- MAE
- MAPE (%)
- R²
- Adjusted R²
- Residuals distribution
- Cook distance
- Feature coefficients / importances
- Cross-validation scores

### Classification (ML)
- Accuracy
- Precision / Recall
- F1-score (macro / weighted)
- Support
- Confusion matrix
- ROC curve + AUC
- PR curve + AUC
- Log-loss
- Class distribution
- Feature importances
- Probabilités prédites
- Calibration curve

### Clustering (ML)
- Silhouette score
- Davies-Bouldin index
- Calinski-Harabasz score
- Inertia (K-Means)
- Cluster centroids
- Taille des clusters
- Variance intra / inter-clusters

### Séries Temporelles
- Paramètres du modèle (p, d, q)
- AIC / AICc / BIC
- Coefficients AR, MA
- P-value par paramètre
- Ljung-Box test
- Residual autocorrelation
- RMSE / MAPE
- Forecast values
- IC 80% / 95%

## Fonctionnalités Avancées

### Gestion des Corrélations
*À implémenter entre étapes 5 et 6*

Permet à l'utilisateur de :
- Visualiser les corrélations > 0.8
- Sélectionner les features à supprimer
- Relancer les analyses sans recommencer

### Export
- **Rapport HTML** : Formaté avec Bootstrap
- **Rapport PDF** : Généré avec ReportLab
- **Code Python** : Reproductible
- **Bundle ZIP** : Tout inclus (données, modèles, rapports, visuels)

### Simulation
- Formulaire dynamique selon features
- Validation des entrées
- Transformation automatique (scaling, encoding)
- Prédiction avec intervalles de confiance
- Explication des contributions

## Design et UX

### Palette de Couleurs
- **Primary** : Gradient violet-bleu (#667eea → #764ba2)
- **Success** : Vert (#28a745)
- **Warning** : Jaune (#ffc107)
- **Danger** : Rouge (#dc3545)
- **Info** : Bleu clair (#17a2b8)

### Composants Visuels
- **Metric Cards** : Cartes avec valeur et label
- **Quality Score Circle** : Cercle coloré selon score
- **Analysis Cards** : Cartes cliquables pour sélection
- **Stepper** : Indicateur de progression horizontal
- **Feature Cards** : Cartes d'explication des étapes

### Responsive
- Mobile first
- Breakpoints Bootstrap 5
- Stepper adaptatif
- Tables scrollables

## Utilisation

### Démarrage
```bash
python manage.py runserver
```

Accéder à : `http://localhost:8000/`

### Interface Classique (backup)
Accessible via : `http://localhost:8000/classic/`

## Technologies

- **Backend** : Django 4.2
- **Frontend** : Bootstrap 5.3, Font Awesome 6.5
- **Data** : Pandas, NumPy
- **ML** : scikit-learn, XGBoost, LightGBM
- **Viz** : Matplotlib, Seaborn, Plotly
- **Stats** : SciPy, statsmodels

## Prochaines Améliorations

1. **Gestion des corrélations** : Interface dédiée entre étapes 5-6
2. **Métriques complètes** : Affichage détaillé dans étape 6
3. **Batch Simulation** : Prédiction multiple CSV
4. **Comparaison de modèles** : Vue côte-à-côte
5. **Explainability** : SHAP values, LIME
6. **Optimisation auto** : AutoML basique
7. **Export Excel** : Avec formatage avancé
8. **Dashboard interactif** : Plotly Dash intégré
9. **API REST** : Pour intégrations externes
10. **Multi-datasets** : Support vraiment multi-bases

## Contribution

Pour contribuer :
1. Fork le repo
2. Créer une branche feature
3. Commit et push
4. Ouvrir une Pull Request

## Licence

MIT License - voir LICENSE pour détails
