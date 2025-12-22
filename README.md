# DataAnalyzer 2.0

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📊 Description

DataAnalyzer 2.0 est une plateforme no-code d'analyse de données professionnelle, équivalente à un notebook Python complet (pandas, scikit-learn, statsmodels) mais avec une interface graphique intuitive.

### ✨ Fonctionnalités principales

- **Chargement de données** : CSV, Excel, JSON
- **Profiling automatique** : Détection de types, métriques de qualité
- **Analyses exploratoires (EDA)** : Statistiques, corrélations, distributions, outliers
- **Modélisation ML** : Régression, classification avec validation scientifique stricte
- **Export complet** : Rapports HTML, code Python, modèles, données
- **Pédagogie intégrée** : Explications à chaque étape

### 🎯 Règles scientifiques strictes

#### RÈGLE 1 : Séparation cible/features
La variable cible ne peut **JAMAIS** être utilisée comme variable explicative.

```python
# ❌ INTERDIT
X = df[features + [target]]

# ✅ CORRECT
X = df[features]
y = df[target]
```

#### RÈGLE 2 : Cohérence des analyses
- Classification → Cible catégorielle → Accuracy, F1, ROC-AUC
- Régression → Cible numérique → R², RMSE, MAE

#### RÈGLE 3 : Transparence totale
Tous les paramètres sont affichés et personnalisables.

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Étapes

```bash
# Cloner le repository
git clone https://github.com/Elm-as/DataAnalyzer2.0.git
cd DataAnalyzer2.0

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

## 📂 Structure du projet

```
DataAnalyzer2.0/
├── app.py                    # Point d'entrée Streamlit
├── requirements.txt          # Dépendances Python
├── README.md
├── data/
│   ├── Titanic-Dataset.csv   # Dataset d'exemple
│   └── uploads/              # Fichiers utilisateurs
├── modules/
│   ├── data_loader.py        # Chargement fichiers
│   ├── data_profiler.py      # Profiling automatique
│   ├── eda.py               # Analyses exploratoires
│   ├── ml_models.py         # Modèles ML
│   ├── time_series.py       # Séries temporelles
│   ├── text_analysis.py     # Analyse texte
│   ├── visualizations.py    # Graphiques
│   └── export.py            # Export rapports
└── utils/
    ├── validation.py        # Validation scientifique
    └── explanations.py      # Textes pédagogiques
```

## 🎓 Utilisation

### 1. Chargement des données

- Uploader un fichier CSV/Excel/JSON
- Ou utiliser le dataset Titanic pré-chargé

### 2. Sélection de la cible

- Choisir la variable à prédire
- Le système détecte automatiquement le type (régression/classification)
- **La cible est automatiquement exclue des features**

### 3. Exploration (EDA)

- Statistiques descriptives
- Corrélations (Pearson/Spearman)
- Distributions avec KDE
- Détection d'anomalies (IQR)
- Analyse catégorielle

### 4. Modélisation

- **Régression** : Linear, Ridge, Lasso, Random Forest, XGBoost, LightGBM
- **Classification** : Logistic, Random Forest, XGBoost, LightGBM
- Métriques automatiques selon le type
- Feature importance
- Validation train/test

### 5. Export

- Rapport HTML professionnel
- Code Python reproductible
- Modèles entraînés (pickle)
- Données transformées
- Session complète (JSON)

## 📖 Exemple avec Titanic

```python
# 1. Charger Titanic-Dataset.csv
# 2. Sélectionner Survived comme cible
#    → Type détecté : Classification binaire
# 3. Features auto-sélectionnées (sans Survived)
# 4. Entraîner Random Forest
#    → Accuracy ~82%
#    → Features importantes : Sex, Age, Pclass
```

## 🧪 Tests de validation

Le système passe ces tests :

1. ✅ Dataset IRIS → Species = cible → Classification uniquement
2. ✅ Dataset Titanic → Survived = cible → Jamais dans features
3. ✅ Séparation stricte X/y → Validation scientifique
4. ✅ Métriques cohérentes avec le type de problème

## 🛠️ Technologies

- **Frontend/Backend** : Streamlit
- **Data Processing** : pandas, numpy
- **ML** : scikit-learn, xgboost, lightgbm
- **Stats** : scipy, statsmodels
- **Visualisation** : matplotlib, seaborn, plotly

## 📊 Captures d'écran

*(À ajouter après le premier déploiement)*

## 🤝 Contribution

Les contributions sont les bienvenues ! Veuillez :

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 License

MIT License - voir le fichier LICENSE pour plus de détails

## 👤 Auteur

Elm-as

## 🙏 Remerciements

- Dataset Titanic : [Kaggle](https://www.kaggle.com/c/titanic)
- Streamlit pour le framework
- Communauté open-source

---

**DataAnalyzer 2.0** - Analyse de données accessible à tous 📊
