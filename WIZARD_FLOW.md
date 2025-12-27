# DataAnalyzer 2.0 - Wizard Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     STEP 0: ACCUEIL 👋                          │
│                                                                  │
│  • Présentation DataAnalyzer 2.0                                │
│  • Guide des 7 étapes avec cartes visuelles                     │
│  • Fonctionnalités clés                                         │
│  • Bouton "Commencer l'analyse"                                 │
│                                                                  │
│            [🚀 Commencer] ──────────────────┐                   │
└─────────────────────────────────────────────┼───────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STEP 1: IMPORT DONNÉES 📥                       │
│                                                                  │
│  • Upload fichier (CSV/Excel/JSON, max 100MB)                  │
│  • Datasets d'exemple (Titanic, IRIS)                          │
│  • Validation format automatique                                │
│  • Métriques rapides (lignes, colonnes, mémoire)               │
│                                                                  │
│  Validation: Données chargées ✓                                │
│                                                                  │
│  [◄ Précédent]         [Suivant: Aperçu ►] ────────┐          │
└─────────────────────────────────────────────────────┼───────────┘
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                STEP 2: APERÇU DONNÉES 👁️                        │
│                                                                  │
│  • Tableau 20 premières lignes                                  │
│  • Types colonnes détectés                                      │
│  • Métriques par type (numériques, catégorielles)              │
│  • Badge N/A par colonne                                        │
│                                                                  │
│  [◄ Précédent]      [Suivant: Qualité ►] ──────────┐          │
└─────────────────────────────────────────────────────┼───────────┘
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│             STEP 3: RAPPORT QUALITÉ 🛡️                          │
│                                                                  │
│  • Score Global (0-100%) avec cercle coloré                     │
│    ○ Excellent (>80%) - Vert                                    │
│    ○ Bon (60-80%) - Bleu                                        │
│    ○ Moyen (40-60%) - Jaune                                     │
│    ○ Faible (<40%) - Rouge                                      │
│                                                                  │
│  • Métriques Complétude                                         │
│    - Lignes complètes                                           │
│    - % valeurs N/A                                              │
│    - Nombre doublons                                            │
│                                                                  │
│  • Détail Colonnes (état par colonne)                           │
│    ✅ OK  ⚠️ Attention  ❌ Problématique                        │
│                                                                  │
│  • ⚠️ Avertissements Automatiques                               │
│    "X lignes dupliquées détectées"                              │
│    "X colonnes problématiques: col1, col2..."                   │
│                                                                  │
│  • 💡 Suggestions Intelligentes                                 │
│    "Envisager de supprimer ces colonnes..."                     │
│    "✅ Qualité suffisante" / "⚠️ Nettoyage recommandé"          │
│                                                                  │
│  [◄ Précédent]   [Suivant: Configuration ►] ────────┐          │
└─────────────────────────────────────────────────────┼───────────┘
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 4: CONFIGURATION ⚙️                            │
│                                                                  │
│  SECTION 1: Sélection Variable Cible                            │
│  • Liste déroulante toutes colonnes                             │
│  • Détection auto type problème (régression/classification)    │
│                                                                  │
│  SECTION 2: Vérification Types                                  │
│  • Tableau: Colonne | Détecté | À Utiliser                     │
│  • Modification manuelle (num/cat/text/date/bool)              │
│                                                                  │
│  SECTION 3: Sélection Features                                  │
│  • Exclusion automatique cible                                  │
│  • Barre recherche + filtrage                                   │
│  • Boutons "Tout sélectionner/désélectionner"                  │
│  • Recommandations (désélectionner IDs, noms)                  │
│                                                                  │
│  Validation: Cible sélectionnée ✓                              │
│  Validation: Cible ≠ Features ✓                                │
│                                                                  │
│  [◄ Précédent]     [Enregistrer Configuration] ─────┐          │
└─────────────────────────────────────────────────────┼───────────┘
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│            STEP 5: SÉLECTION ANALYSES 📊                         │
│                                                                  │
│  Métriques: Lignes | Colonnes | Numériques | Temps Estimé     │
│                                                                  │
│  📊 ANALYSES DE BASE (JavaScript)                               │
│  ┌─────────────────────────────────────────┐                  │
│  │ ✅ Statistiques descriptives  (~2s)     │ [Activé]         │
│  │ ✅ Corrélations              (~3s)     │ [Activé]         │
│  │ ✅ Distributions             (~4s)     │ [Activé]         │
│  │ ✅ Détection anomalies       (~3s)     │ [Activé]         │
│  │ ✅ Analyse catégorielle      (~3s)     │ [Activé]         │
│  └─────────────────────────────────────────┘                  │
│                                                                  │
│  🧠 ANALYSES AVANCÉES (Python ML/DL)                            │
│  ┌─────────────────────────────────────────┐                  │
│  │ 🧠 Régression ML           (~10s)       │ [Si cible num]  │
│  │ 🧠 Classification ML       (~12s)       │ [Si cible cat]  │
│  │ 🧠 Clustering              (~12s)       │ [Si ≥2 num]     │
│  │ 🧠 Séries temporelles      (~20s)       │ [Si date+num]   │
│  └─────────────────────────────────────────┘                  │
│                                                                  │
│  ★ BONUS: Gestion Corrélations 🔗                               │
│  [📊 Gérer les Corrélations] ────────┐                         │
│                                       │                          │
│  Validation: ≥1 analyse sélectionnée ✓│                         │
│                                       │                          │
│  [◄ Précédent]  [Enregistrer Sélection]                        │
│                                       │                          │
│                 [▶ Lancer les Analyses] ─────────┐              │
└───────────────────────────────────────┼──────────┼──────────────┘
                                        │          │
                    ┌───────────────────┘          │
                    │                              │
                    ▼                              ▼
    ┌───────────────────────────┐    ┌────────────────────────────┐
    │  STEP 5b: CORRÉLATIONS 🔗 │    │    STEP 6: RÉSULTATS 📈    │
    │                           │    │                            │
    │ • Détection corr > 0.7    │    │  📑 ONGLET RÉSUMÉ         │
    │ • Tableau paires corrélées│    │  • Liste analyses ✅       │
    │ • Action: Supprimer var1  │    │  • Meilleur modèle 🏆     │
    │   ou var2                 │    │  • Score + temps          │
    │ • Conseils pour choisir   │    │                            │
    │ • Prévention multi-select │    │  📊 ONGLETS INDIVIDUELS   │
    │ • Matrice visualisation   │    │  • Statistiques            │
    │                           │    │  • Corrélations            │
    │ [◄ Retour]  [Appliquer ►]│    │  • Distributions           │
    │             ↓             │    │  • Anomalies               │
    └─────────────┼─────────────┘    │  • Catégorielles           │
                  │                   │  • Régression/Classif      │
                  └──────┐            │  • Clustering              │
                         │            │  • Séries temporelles      │
                         ▼            │                            │
              [Retour Step 5]         │  💾 EXPORTS                │
                                      │  • Rapport HTML            │
                                      │  • Rapport PDF             │
                                      │  • Code Python             │
                                      │  • Bundle ZIP              │
                                      │                            │
                                      │  [◄ Précédent]             │
                                      │  [Suivant: Simulation ►]──┐│
                                      └────────────────────────────┼┘
                                                                   │
                                                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│          STEP 7: SIMULATION & PRÉDICTION 🎯                      │
│                                                                  │
│  Si modèle ML disponible:                                       │
│                                                                  │
│  • Formulaire Dynamique (auto-généré selon features)           │
│    - Types intelligents (numérique vs texte)                   │
│    - Exclusion automatique cible                               │
│    - Placeholders contextuels                                  │
│                                                                  │
│  • Résultat Prédiction                                          │
│    ┌─────────────────────────────────┐                         │
│    │  🎯 Prédiction pour [TARGET]    │                         │
│    │                                  │                         │
│    │  Valeur prédite:                │                         │
│    │  ┌────────────────────────────┐ │                         │
│    │  │       [VALEUR]             │ │                         │
│    │  └────────────────────────────┘ │                         │
│    │                                  │                         │
│    │  Probabilités (si classif):     │                         │
│    │  Classe 0: 0.234                │                         │
│    │  Classe 1: 0.766                │                         │
│    └─────────────────────────────────┘                         │
│                                                                  │
│  • Interprétation                                               │
│    "La prédiction est produite à partir des features           │
│     sélectionnées, sans jamais demander la cible."             │
│                                                                  │
│  • Informations                                                 │
│    ✓ Transformations auto (scaling, encoding)                  │
│    ✓ Imputation automatique                                    │
│    ⚠️ Attention valeurs hors domaine                           │
│                                                                  │
│  [◄ Précédent]      [✓ Terminer l'Analyse]                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                        [Retour Accueil]
                        ou [Nouvelle Analyse]
```

## 🔄 Flux de Validation

```
Step 0 ──► Step 1 (toujours accessible)
         ✓
Step 1 ──► Step 2 (si données chargées)
         ✓
Step 2 ──► Step 3 (libre)
         ✓
Step 3 ──► Step 4 (libre)
         ✓
Step 4 ──► Step 5 (si cible sélectionnée ET cible ≠ features)
         ✓
Step 5 ──┬─► Step 5b (optionnel: gestion corrélations)
         │   ✓
         │   Step 5b ──► Step 5 (retour après application)
         │
         └─► Step 6 (si ≥1 analyse sélectionnée ET analyses lancées)
         ✓
Step 6 ──► Step 7 (si modèle ML entraîné)
         ✓
Step 7 ──► Fin (Terminer ou Recommencer)
```

## 🎨 Éléments Visuels Clés

### Stepper Horizontal
```
  ①────②────③────④────⑤────⑥────⑦
Import Aperçu Qualité Config Analyses Résultats Simulation

États:
• Futur:     Gris, opacité 50%
• Actif:     Blanc, scale 1.1, shadow
• Complété:  Vert avec ✓
```

### Quality Score Circle
```
     ┌─────────┐
     │   85%   │  ← Couleur selon score
     │ Excellent│  
     └─────────┘
     
Couleurs:
• >80%:  Vert (Excellent)
• 60-80: Bleu (Bon)
• 40-60: Jaune (Moyen)
• <40:   Rouge (Faible)
```

### Analysis Card (Sélectionnable)
```
┌────────────────────────────────┐
│ ✅  Statistiques descriptives  │
│     ~2s                        │
│     Moyenne, médiane, écart... │
│                                │
│     [Activé ✓]                 │
└────────────────────────────────┘

États:
• Normal:     Border grise
• Hover:      Border bleue + shadow
• Sélectionnée: Border bleue + bg bleu clair
• Désactivée: Opacité 50% + cursor not-allowed
```

### Navigation Buttons
```
┌─────────────────────────────────────────┐
│  [◄ Précédent]         [Suivant ►]     │  ← Fixed bottom bar
└─────────────────────────────────────────┘

Variantes:
• Suivant désactivé si validation échoue
• Suivant remplacé par action spécifique (Lancer Analyses, Terminer)
• Précédent masqué sur Step 1
```

## 🔑 Clés de Session Django

```python
wizard_step: int                    # 0-7
wizard_completed_steps: List[int]   # [0, 1, 2, ...]
wizard_selected_analyses: List[str] # ['descriptive', 'correlation', ...]
wizard_quality_warnings: List[str]  # ["X lignes dupliquées", ...]
wizard_quality_suggestions: List[str] # ["Envisager de supprimer...", ...]
wizard_analysis_results: Dict       # {'descriptive': {...}, ...}

# Hérité de l'ancien système:
SESSION_KEY_TARGET: str             # Nom colonne cible
SESSION_KEY_FEATURES: List[str]     # Liste features sélectionnées
SESSION_KEY_MANUAL_TYPES: Dict      # {'col1': 'numeric', ...}
SESSION_KEY_MODEL_BUNDLE_PATH: str  # Chemin modèle entraîné
```

---

*Document généré automatiquement*
*DataAnalyzer 2.0 - Assistant Intelligent*
