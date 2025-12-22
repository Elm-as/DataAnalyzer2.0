# 🚀 Quick Start Guide - DataAnalyzer 2.0

## Installation rapide

```bash
# 1. Cloner le repository
git clone https://github.com/Elm-as/DataAnalyzer2.0.git
cd DataAnalyzer2.0

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Tester l'installation
python test_validation.py
```

Si tous les tests passent (✅), l'installation est réussie !

## Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvrira dans votre navigateur à l'adresse `http://localhost:8501`

## Premier usage avec Titanic

### Étape 1: Charger les données
1. Aller dans l'onglet "📂 1. Chargement & Préparation"
2. Sélectionner "📊 Dataset d'exemple (Titanic)"
3. Cliquer sur "🚀 Charger Titanic Dataset"

### Étape 2: Sélectionner la cible
1. Dans la section "1.4 🎯 Sélection de la variable cible"
2. Choisir "Survived" dans le menu déroulant
3. Cliquer sur "✅ Confirmer la cible"
4. ✅ Le système détecte automatiquement: "Classification binaire"
5. ✅ Les features sont automatiquement sélectionnées (Survived est exclue)

### Étape 3: Explorer les données (EDA)
1. Aller dans l'onglet "🔍 2. Exploration (EDA)"
2. Essayer différentes analyses:
   - **Statistiques descriptives**: Vue d'ensemble des variables numériques
   - **Corrélations**: Relation entre Age, Fare, Pclass, etc.
   - **Distributions**: Voir la distribution de l'âge
   - **Détection d'anomalies**: Trouver les outliers dans Fare
   - **Analyse catégorielle**: Fréquences de Sex, Embarked

### Étape 4: Entraîner des modèles
1. Aller dans l'onglet "🤖 3. Modélisation (ML)"
2. Configuration:
   - Taille test: 20%
   - Random seed: 42
   - Modèles: Logistic, Random Forest, XGBoost
3. Cliquer sur "🚀 Entraîner les modèles"
4. Résultats attendus:
   - Meilleur modèle: Random Forest ou XGBoost
   - Accuracy: ~80-82%
   - Features importantes: Sex, Pclass, Fare, Age

### Étape 5: Exporter les résultats
1. Aller dans l'onglet "💾 6. Export & Rapports"
2. Options disponibles:
   - Exporter les données (CSV/Excel/JSON)
   - Sauvegarder la session (JSON)
   - Générer un rapport HTML

## ⚠️ Règles importantes

### RÈGLE 1: Séparation cible/features (STRICTEMENT APPLIQUÉE)
```python
# ❌ JAMAIS faire cela
X = df[features + [target]]  # INTERDIT!

# ✅ TOUJOURS faire cela
X = df[features]  # Sans la cible
y = df[target]    # Cible séparée
```

**Le système empêche automatiquement d'inclure la cible dans les features.**

Si vous tentez d'inclure la cible dans les features:
- ⚠️ Message d'erreur: "La variable cible ne peut pas être utilisée comme variable explicative"
- ❌ L'entraînement sera refusé

### RÈGLE 2: Types de problèmes
- **Cible numérique** → Régression (R², RMSE, MAE)
- **Cible catégorielle (2 classes)** → Classification binaire (Accuracy, F1, ROC-AUC)
- **Cible catégorielle (>2 classes)** → Classification multiclasse (Accuracy, F1)

### RÈGLE 3: Métriques de qualité
Avant toute analyse, vérifier:
- ✅ Valeurs manquantes < 20%
- ✅ Doublons < 5%
- ✅ Distribution équilibrée des classes (classification)

## 📊 Exemples de résultats avec Titanic

### Corrélations significatives
- Fare ↔ Pclass: -0.55 (forte corrélation négative)
- Survived ↔ Fare: 0.26 (corrélation positive modérée)

### Outliers détectés
- Fare: Quelques billets très chers (>500)
- Age: Personnes très âgées (>70)

### Meilleur modèle
- Algorithm: Random Forest
- Accuracy: ~82%
- Features importantes:
  1. Sex (le plus important)
  2. Pclass
  3. Fare
  4. Age

### Interprétation
Les femmes de 1ère classe ont les meilleures chances de survie.

## 🐛 Résolution de problèmes

### Erreur: "Module not found"
```bash
pip install -r requirements.txt
```

### Erreur: "No module named 'streamlit'"
```bash
pip install streamlit
```

### L'application ne se lance pas
Vérifier que le port 8501 n'est pas déjà utilisé:
```bash
streamlit run app.py --server.port 8502
```

### Erreur de mémoire avec gros dataset
Activer l'échantillonnage dans l'interface (option disponible pour >10,000 lignes)

## 📚 Documentation complète

Pour plus de détails, consulter:
- `README.md`: Documentation complète
- `test_validation.py`: Tests de validation
- Code source des modules dans `modules/` et `utils/`

## 🆘 Support

Pour toute question ou bug:
1. Vérifier que `test_validation.py` passe tous les tests
2. Consulter les messages d'erreur détaillés dans l'interface
3. Ouvrir une issue sur GitHub

---

**Bon analyse! 📊**
