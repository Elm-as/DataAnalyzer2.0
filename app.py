"""
DataAnalyzer 2.0 - Main Application
Plateforme no-code d'analyse de données professionnelle
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent))

# Import des modules
from modules.data_loader import load_data, save_uploaded_file, get_data_preview
from modules.data_profiler import (
    profile_dataframe, detect_column_type, get_column_recommendations,
    suggest_target_variables, get_data_quality_score
)
from modules.eda import (
    descriptive_statistics, correlation_analysis, detect_outliers,
    categorical_analysis, distribution_analysis
)
from modules.ml_models import (
    train_regression_model, train_classification_model, clustering_analysis
)
from modules.time_series import analyze_time_series
from modules.text_analysis import analyze_text
from modules import visualizations as viz
from modules.export import (
    generate_python_code, export_model, export_data,
    generate_html_report, save_session
)
from utils.validation import (
    validate_target_not_in_features, validate_analysis_requirements,
    detect_problem_type, get_recommended_metrics
)
from utils.explanations import (
    get_method_explanation, get_titanic_example, get_tips
)

# Configuration de la page
st.set_page_config(
    page_title="DataAnalyzer 2.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation de la session
if 'df' not in st.session_state:
    st.session_state.df = None
if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'features' not in st.session_state:
    st.session_state.features = []
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def main():
    """Application principale"""
    
    # En-tête
    st.markdown('<h1 class="main-header">📊 DataAnalyzer 2.0</h1>', unsafe_allow_html=True)
    st.markdown("### Plateforme no-code d'analyse de données professionnelle")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/3498db/ffffff?text=DataAnalyzer", use_column_width=True)
        st.markdown("---")
        st.markdown("### 🎯 Navigation")
        
        # Information sur les données chargées
        if st.session_state.df is not None:
            st.success(f"✅ Données chargées: {st.session_state.df.shape[0]} lignes × {st.session_state.df.shape[1]} colonnes")
            if st.session_state.target:
                st.info(f"🎯 Cible: {st.session_state.target}")
                st.info(f"📝 Type: {st.session_state.problem_type}")
    
    # Onglets principaux
    tabs = st.tabs([
        "📂 1. Chargement & Préparation",
        "🔍 2. Exploration (EDA)",
        "🤖 3. Modélisation (ML)",
        "📊 4. Évaluation & Diagnostics",
        "🎯 5. Simulation & Prédiction",
        "💾 6. Export & Rapports"
    ])
    
    # ========== ONGLET 1: CHARGEMENT & PRÉPARATION ==========
    with tabs[0]:
        st.header("📂 Chargement et Préparation des Données")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("1.1 Source des données")
            
            data_source = st.radio(
                "Choisir la source:",
                ["📊 Dataset d'exemple (Titanic)", "📤 Uploader un fichier"],
                horizontal=True
            )
            
            if data_source == "📊 Dataset d'exemple (Titanic)":
                if st.button("🚀 Charger Titanic Dataset", type="primary"):
                    titanic_path = "data/Titanic-Dataset.csv"
                    if os.path.exists(titanic_path):
                        df, error = load_data(titanic_path)
                        if df is not None:
                            st.session_state.df = df
                            st.session_state.profile = profile_dataframe(df)
                            st.success("✅ Titanic dataset chargé avec succès!")
                            st.rerun()
                        else:
                            st.error(f"❌ Erreur: {error}")
                    else:
                        st.error("❌ Fichier Titanic-Dataset.csv introuvable")
            
            else:
                uploaded_file = st.file_uploader(
                    "Choisir un fichier",
                    type=['csv', 'xlsx', 'xls', 'json'],
                    help="Formats supportés: CSV, Excel, JSON"
                )
                
                if uploaded_file is not None:
                    separator = st.selectbox("Séparateur CSV:", [',', ';', '\t'])
                    
                    if st.button("📥 Charger le fichier", type="primary"):
                        # Sauvegarder le fichier
                        file_path, error = save_uploaded_file(uploaded_file)
                        
                        if file_path:
                            df, error = load_data(file_path, separator)
                            if df is not None:
                                st.session_state.df = df
                                st.session_state.profile = profile_dataframe(df)
                                st.success(f"✅ Fichier chargé: {uploaded_file.name}")
                                st.rerun()
                            else:
                                st.error(f"❌ Erreur: {error}")
                        else:
                            st.error(f"❌ Erreur sauvegarde: {error}")
        
        with col2:
            if st.session_state.df is not None:
                st.subheader("📋 Informations rapides")
                st.metric("Lignes", st.session_state.df.shape[0])
                st.metric("Colonnes", st.session_state.df.shape[1])
                
                if st.session_state.profile:
                    score, level = get_data_quality_score(st.session_state.profile)
                    st.metric("Qualité des données", f"{score}%", level)
        
        # Affichage des données
        if st.session_state.df is not None:
            st.markdown("---")
            st.subheader("1.2 Aperçu des données")
            
            tab_head, tab_tail, tab_sample = st.tabs(["👆 Premières lignes", "👇 Dernières lignes", "🎲 Échantillon"])
            
            with tab_head:
                n_rows = st.slider("Nombre de lignes:", 5, 50, 10, key="head")
                st.dataframe(st.session_state.df.head(n_rows), use_container_width=True)
            
            with tab_tail:
                n_rows = st.slider("Nombre de lignes:", 5, 50, 10, key="tail")
                st.dataframe(st.session_state.df.tail(n_rows), use_container_width=True)
            
            with tab_sample:
                n_rows = st.slider("Nombre de lignes:", 5, 50, 10, key="sample")
                st.dataframe(st.session_state.df.sample(min(n_rows, len(st.session_state.df))), use_container_width=True)
            
            # Profil des données
            st.markdown("---")
            st.subheader("1.3 Profil des données")
            
            if st.session_state.profile:
                profile = st.session_state.profile
                
                # Métriques de qualité
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Valeurs manquantes", f"{profile['quality_metrics']['pct_missing']:.1f}%")
                with col2:
                    st.metric("Doublons", f"{profile['quality_metrics']['pct_duplicates']:.1f}%")
                with col3:
                    st.metric("Lignes complètes", f"{profile['quality_metrics']['pct_complete_rows']:.1f}%")
                with col4:
                    memory_mb = profile['memory_usage']
                    st.metric("Mémoire", f"{memory_mb:.1f} MB")
                
                # Types de colonnes
                st.markdown("#### 📊 Types détectés")
                type_summary = profile['type_summary']
                cols = st.columns(len(type_summary))
                for i, (col_type, count) in enumerate(type_summary.items()):
                    with cols[i]:
                        st.metric(col_type.title(), count)
                
                # Détail des colonnes
                st.markdown("#### 📋 Détail des colonnes")
                
                columns_df = pd.DataFrame([
                    {
                        'Colonne': col,
                        'Type': info['type'],
                        'Manquantes': f"{info['pct_missing']:.1f}%",
                        'Uniques': info['n_unique'],
                        'Dtype': info['dtype']
                    }
                    for col, info in profile['columns'].items()
                ])
                st.dataframe(columns_df, use_container_width=True)
                
                # Recommandations
                recommendations = get_column_recommendations(st.session_state.df, profile)
                if recommendations:
                    st.markdown("#### ⚠️ Recommandations")
                    for col, recs in recommendations.items():
                        with st.expander(f"🔍 {col}"):
                            for rec in recs:
                                st.markdown(f"- {rec}")
            
            # Sélection de la cible
            st.markdown("---")
            st.subheader("1.4 🎯 Sélection de la variable cible")
            
            st.markdown(get_titanic_example('target_selection'))
            
            # Suggestions
            if st.session_state.profile:
                candidates = suggest_target_variables(st.session_state.df, st.session_state.profile)
                st.info(f"💡 Variables candidates suggérées: {', '.join(candidates[:5])}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                target = st.selectbox(
                    "Sélectionner la variable cible:",
                    ["Aucune"] + list(st.session_state.df.columns),
                    index=0 if st.session_state.target is None else list(st.session_state.df.columns).index(st.session_state.target) + 1
                )
                
                if target != "Aucune" and st.button("✅ Confirmer la cible", type="primary"):
                    st.session_state.target = target
                    problem_type, description = detect_problem_type(st.session_state.df[target])
                    st.session_state.problem_type = description
                    
                    # Sélectionner automatiquement toutes les features SAUF la cible
                    all_features = [col for col in st.session_state.df.columns if col != target]
                    st.session_state.features = all_features
                    
                    st.success(f"✅ Cible définie: {target}")
                    st.success(f"📊 Type détecté: {description}")
                    st.success(f"✅ Features automatiquement sélectionnées (cible exclue)")
                    st.rerun()
            
            with col2:
                if st.session_state.target:
                    st.markdown("##### 📊 Informations cible")
                    target_col = st.session_state.df[st.session_state.target]
                    st.metric("Valeurs uniques", target_col.nunique())
                    st.metric("Manquantes", f"{(target_col.isna().sum()/len(target_col)*100):.1f}%")
                    
                    if target_col.dtype in ['object', 'category'] or target_col.nunique() < 20:
                        st.markdown("##### Distribution")
                        dist = target_col.value_counts().head(10)
                        st.bar_chart(dist)
            
            # Sélection des features
            if st.session_state.target:
                st.markdown("---")
                st.subheader("1.5 📝 Sélection des variables explicatives (Features)")
                
                # RÈGLE STRICTE: La cible ne doit PAS être dans les features
                available_features = [col for col in st.session_state.df.columns if col != st.session_state.target]
                
                st.markdown(f"""
                <div class="warning-box">
                    <strong>⚠️ RÈGLE IMPORTANTE:</strong><br>
                    La variable cible <strong>{st.session_state.target}</strong> est automatiquement exclue des features.<br>
                    Elle ne peut PAS être utilisée comme variable explicative.
                </div>
                """, unsafe_allow_html=True)
                
                selected_features = st.multiselect(
                    "Sélectionner les features:",
                    available_features,
                    default=st.session_state.features if st.session_state.features else available_features
                )
                
                # Validation
                is_valid, message = validate_target_not_in_features(selected_features, st.session_state.target)
                
                if not is_valid:
                    st.error(message)
                else:
                    if st.button("💾 Sauvegarder la sélection"):
                        st.session_state.features = selected_features
                        st.success(f"✅ {len(selected_features)} features sélectionnées")
                        st.rerun()
                
                if st.session_state.features:
                    st.info(f"✅ Features actuelles: {len(st.session_state.features)} variables")
                    with st.expander("📋 Voir la liste"):
                        st.write(st.session_state.features)
    
    # ========== ONGLET 2: EXPLORATION (EDA) ==========
    with tabs[1]:
        st.header("🔍 Exploration des Données (EDA)")
        
        if st.session_state.df is None:
            st.warning("⚠️ Veuillez d'abord charger des données dans l'onglet 1")
        else:
            df = st.session_state.df
            
            # Analyses disponibles
            st.subheader("Analyses disponibles")
            
            analysis_type = st.selectbox(
                "Choisir une analyse:",
                [
                    "Statistiques descriptives",
                    "Corrélations",
                    "Distributions",
                    "Détection d'anomalies",
                    "Analyse catégorielle"
                ]
            )
            
            # Paramètres selon l'analyse
            st.markdown("---")
            st.subheader("⚙️ Paramètres")
            
            if analysis_type == "Statistiques descriptives":
                st.markdown(get_method_explanation('descriptive_stats')['method'])
                
                if st.button("▶️ Exécuter l'analyse", type="primary"):
                    with st.spinner("Analyse en cours..."):
                        results = descriptive_statistics(df)
                        
                        if results['success']:
                            st.success(f"✅ Analyse terminée en {results['execution_time']:.2f}s")
                            
                            # Afficher les résultats
                            stats_df = pd.DataFrame(results['results']['statistics']).T
                            st.dataframe(stats_df, use_container_width=True)
                            
                            # Explication
                            with st.expander("📚 Interprétation"):
                                st.markdown(results['explanations']['interpretation'])
                        else:
                            st.error(f"❌ Erreur: {results.get('error', 'Inconnue')}")
            
            elif analysis_type == "Corrélations":
                col1, col2 = st.columns(2)
                with col1:
                    method = st.selectbox("Méthode:", ["pearson", "spearman"])
                with col2:
                    threshold = st.slider("Seuil minimal:", 0.0, 1.0, 0.0, 0.05)
                
                st.markdown(get_method_explanation('correlation')['method'])
                st.markdown(get_titanic_example('correlation'))
                
                if st.button("▶️ Exécuter l'analyse", type="primary"):
                    with st.spinner("Calcul des corrélations..."):
                        results = correlation_analysis(df, params={'method': method, 'threshold': threshold})
                        
                        if results['success']:
                            st.success(f"✅ Analyse terminée en {results['execution_time']:.2f}s")
                            
                            # Corrélations fortes
                            if results['results']['strong_correlations']:
                                st.markdown("#### 🔥 Corrélations significatives")
                                corr_df = pd.DataFrame(results['results']['strong_correlations'])
                                st.dataframe(corr_df, use_container_width=True)
                            else:
                                st.info(f"Aucune corrélation >= {threshold} trouvée")
                            
                            # Matrice de corrélation
                            st.markdown("#### 📊 Matrice de corrélation")
                            corr_matrix = pd.DataFrame(results['results']['correlation_matrix'])
                            fig = viz.plot_correlation_matrix(corr_matrix, method, threshold)
                            st.pyplot(fig)
                            
                            # Explication
                            with st.expander("📚 Interprétation"):
                                st.markdown(results['explanations']['interpretation'])
                        else:
                            st.error(f"❌ Erreur: {results.get('error', 'Inconnue')}")
            
            elif analysis_type == "Distributions":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_cols:
                    st.warning("Aucune colonne numérique trouvée")
                else:
                    selected_col = st.selectbox("Sélectionner une variable:", numeric_cols)
                    bins = st.slider("Nombre de bins:", 10, 100, 30)
                    show_kde = st.checkbox("Afficher la courbe KDE", value=True)
                    
                    if st.button("▶️ Exécuter l'analyse", type="primary"):
                        with st.spinner("Analyse en cours..."):
                            results = distribution_analysis(df, columns=[selected_col], params={'bins': bins})
                            
                            if results['success']:
                                st.success(f"✅ Analyse terminée en {results['execution_time']:.2f}s")
                                
                                # Statistiques
                                dist_info = results['results']['distribution_summary'][selected_col]
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Moyenne", f"{dist_info['mean']:.2f}")
                                with col2:
                                    st.metric("Médiane", f"{dist_info['median']:.2f}")
                                with col3:
                                    st.metric("Écart-type", f"{dist_info['std']:.2f}")
                                with col4:
                                    st.metric("Asymétrie", f"{dist_info['skewness']:.2f}")
                                
                                # Graphique
                                fig = viz.plot_distribution(df[selected_col], bins=bins, show_kde=show_kde)
                                st.pyplot(fig)
                                
                                # Interprétation
                                with st.expander("📚 Interprétation"):
                                    st.markdown(results['explanations']['interpretation'])
                                    if dist_info.get('is_normal') is not None:
                                        if dist_info['is_normal']:
                                            st.success("✅ Distribution approximativement normale")
                                        else:
                                            st.warning("⚠️ Distribution non normale")
                            else:
                                st.error(f"❌ Erreur: {results.get('error', 'Inconnue')}")
            
            elif analysis_type == "Détection d'anomalies":
                iqr_multiplier = st.slider("Multiplicateur IQR:", 1.0, 3.0, 1.5, 0.1)
                
                st.markdown(get_method_explanation('anomaly_detection')['method'])
                st.markdown(get_titanic_example('anomaly_detection'))
                
                if st.button("▶️ Exécuter l'analyse", type="primary"):
                    with st.spinner("Détection des outliers..."):
                        results = detect_outliers(df, params={'iqr_multiplier': iqr_multiplier})
                        
                        if results['success']:
                            st.success(f"✅ Analyse terminée en {results['execution_time']:.2f}s")
                            
                            # Résumé
                            total_outliers = results['results']['total_outlier_rows']
                            pct_outliers = results['results']['pct_outlier_rows']
                            
                            if total_outliers > 0:
                                st.warning(f"⚠️ {total_outliers} lignes avec outliers ({pct_outliers:.1f}%)")
                                
                                # Détail par colonne
                                outliers_data = []
                                for col, info in results['results']['outliers_by_column'].items():
                                    if info['n_outliers'] > 0:
                                        outliers_data.append({
                                            'Colonne': col,
                                            'Nombre': info['n_outliers'],
                                            'Pourcentage': f"{info['pct_outliers']:.1f}%",
                                            'Borne inf': info['lower_bound'],
                                            'Borne sup': info['upper_bound']
                                        })
                                
                                outliers_df = pd.DataFrame(outliers_data)
                                st.dataframe(outliers_df, use_container_width=True)
                                
                                # Boxplots
                                numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns]
                                if len(numeric_cols) > 0:
                                    fig = viz.plot_boxplot(df, numeric_cols[:5])  # Max 5 colonnes
                                    st.pyplot(fig)
                            else:
                                st.success("✅ Aucun outlier détecté")
                            
                            # Explication
                            with st.expander("📚 Interprétation"):
                                st.markdown(results['explanations']['interpretation'])
                        else:
                            st.error(f"❌ Erreur: {results.get('error', 'Inconnue')}")
            
            elif analysis_type == "Analyse catégorielle":
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                if not cat_cols:
                    st.warning("Aucune colonne catégorielle trouvée")
                else:
                    if st.button("▶️ Exécuter l'analyse", type="primary"):
                        with st.spinner("Analyse en cours..."):
                            results = categorical_analysis(df)
                            
                            if results['success']:
                                st.success(f"✅ Analyse terminée en {results['execution_time']:.2f}s")
                                
                                # Afficher pour chaque colonne
                                for col in results['results']['columns_analyzed']:
                                    with st.expander(f"📊 {col}"):
                                        info = results['results']['categorical_summary'][col]
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Catégories", info['n_unique'])
                                        with col2:
                                            st.metric("Entropie", f"{info['entropy']:.2f}")
                                        with col3:
                                            st.metric("Manquantes", f"{info['missing_pct']:.1f}%")
                                        
                                        # Top values
                                        st.markdown("**Top 10 valeurs:**")
                                        top_df = pd.DataFrame(list(info['top_values'].items()), 
                                                            columns=['Valeur', 'Fréquence'])
                                        st.dataframe(top_df, use_container_width=True)
                                        
                                        # Graphique
                                        fig = viz.plot_categorical_distribution(df[col], top_n=10)
                                        st.pyplot(fig)
                                
                                # Explication
                                with st.expander("📚 Interprétation"):
                                    st.markdown(results['explanations']['interpretation'])
                            else:
                                st.error(f"❌ Erreur: {results.get('error', 'Inconnue')}")
    
    # ========== ONGLET 3: MODÉLISATION ==========
    with tabs[2]:
        st.header("🤖 Modélisation Machine Learning")
        
        if st.session_state.df is None:
            st.warning("⚠️ Veuillez d'abord charger des données")
        elif st.session_state.target is None:
            st.warning("⚠️ Veuillez d'abord sélectionner une variable cible dans l'onglet 1")
        elif not st.session_state.features:
            st.warning("⚠️ Veuillez sélectionner des features dans l'onglet 1")
        else:
            df = st.session_state.df
            target = st.session_state.target
            features = st.session_state.features
            
            # Vérification de sécurité
            is_valid, message = validate_target_not_in_features(features, target)
            if not is_valid:
                st.error(message)
                st.stop()
            
            st.success(f"✅ Cible: {target} | Features: {len(features)} variables")
            st.info(f"📊 Type de problème: {st.session_state.problem_type}")
            
            # Déterminer le type de modélisation
            problem_type, _ = detect_problem_type(df[target])
            
            if problem_type == 'regression':
                st.subheader("📈 Régression")
                st.markdown(get_method_explanation('regression')['method'])
                
                # Paramètres
                col1, col2, col3 = st.columns(3)
                with col1:
                    test_size = st.slider("Taille test (%)", 10, 40, 20) / 100
                with col2:
                    random_state = st.number_input("Random seed", 0, 1000, 42)
                with col3:
                    scale = st.checkbox("Standardiser", value=True)
                
                models = st.multiselect(
                    "Modèles à entraîner:",
                    ['linear', 'ridge', 'lasso', 'random_forest', 'xgboost', 'lightgbm'],
                    default=['linear', 'random_forest', 'xgboost']
                )
                
                if st.button("🚀 Entraîner les modèles", type="primary"):
                    with st.spinner("Entraînement en cours..."):
                        params = {
                            'test_size': test_size,
                            'random_state': random_state,
                            'scale': scale,
                            'models': models
                        }
                        
                        results = train_regression_model(df, target, features, params)
                        
                        if results['success']:
                            st.success(f"✅ Entraînement terminé en {results['execution_time']:.2f}s")
                            
                            # Meilleur modèle
                            best_model = results['results']['best_model']
                            best_score = results['results']['best_score']
                            
                            st.markdown(f"""
                            <div class="success-box">
                                <strong>🏆 Meilleur modèle: {best_model}</strong><br>
                                R² Score: {best_score:.3f}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Comparaison des modèles
                            st.markdown("#### 📊 Comparaison des modèles")
                            
                            comparison_data = []
                            for model_name, model_results in results['results']['models'].items():
                                comparison_data.append({
                                    'Modèle': model_name,
                                    'R² (train)': f"{model_results['train_metrics']['r2']:.3f}",
                                    'R² (test)': f"{model_results['test_metrics']['r2']:.3f}",
                                    'RMSE (test)': f"{model_results['test_metrics']['rmse']:.3f}",
                                    'MAE (test)': f"{model_results['test_metrics']['mae']:.3f}"
                                })
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True)
                            
                            # Feature importance du meilleur modèle
                            if results['results']['models'][best_model]['feature_importance']:
                                st.markdown("#### 🎯 Importance des features (meilleur modèle)")
                                
                                importance = results['results']['models'][best_model]['feature_importance']
                                fig = viz.plot_feature_importance(importance, top_n=15)
                                st.pyplot(fig)
                            
                            # Code Python
                            with st.expander("💻 Code Python généré"):
                                st.code(results['python_code'], language='python')
                            
                            # Explication
                            with st.expander("📚 Interprétation"):
                                st.markdown(results['explanations']['interpretation'])
                        else:
                            st.error(f"❌ Erreur: {results.get('error', 'Inconnue')}")
            
            elif problem_type in ['binary_classification', 'multiclass_classification']:
                st.subheader("🎯 Classification")
                st.markdown(get_method_explanation('classification')['method'])
                st.markdown(get_titanic_example('classification'))
                
                # Paramètres
                col1, col2, col3 = st.columns(3)
                with col1:
                    test_size = st.slider("Taille test (%)", 10, 40, 20) / 100
                with col2:
                    random_state = st.number_input("Random seed", 0, 1000, 42)
                with col3:
                    scale = st.checkbox("Standardiser", value=True)
                
                models = st.multiselect(
                    "Modèles à entraîner:",
                    ['logistic', 'random_forest', 'xgboost', 'lightgbm'],
                    default=['logistic', 'random_forest', 'xgboost']
                )
                
                if st.button("🚀 Entraîner les modèles", type="primary"):
                    with st.spinner("Entraînement en cours..."):
                        params = {
                            'test_size': test_size,
                            'random_state': random_state,
                            'scale': scale,
                            'models': models
                        }
                        
                        results = train_classification_model(df, target, features, params)
                        
                        if results['success']:
                            st.success(f"✅ Entraînement terminé en {results['execution_time']:.2f}s")
                            
                            # Meilleur modèle
                            best_model = results['results']['best_model']
                            best_score = results['results']['best_score']
                            
                            st.markdown(f"""
                            <div class="success-box">
                                <strong>🏆 Meilleur modèle: {best_model}</strong><br>
                                F1-Score: {best_score:.3f}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Comparaison des modèles
                            st.markdown("#### 📊 Comparaison des modèles")
                            
                            comparison_data = []
                            for model_name, model_results in results['results']['models'].items():
                                row = {
                                    'Modèle': model_name,
                                    'Accuracy (test)': f"{model_results['test_metrics']['accuracy']:.3f}",
                                    'Precision (test)': f"{model_results['test_metrics']['precision']:.3f}",
                                    'Recall (test)': f"{model_results['test_metrics']['recall']:.3f}",
                                    'F1-Score (test)': f"{model_results['test_metrics']['f1']:.3f}"
                                }
                                if 'roc_auc' in model_results['test_metrics']:
                                    row['ROC-AUC'] = f"{model_results['test_metrics']['roc_auc']:.3f}"
                                comparison_data.append(row)
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True)
                            
                            # Matrice de confusion du meilleur modèle
                            st.markdown("#### 🎯 Matrice de confusion (meilleur modèle)")
                            cm = np.array(results['results']['models'][best_model]['confusion_matrix'])
                            class_labels = results['results']['class_labels']
                            fig = viz.plot_confusion_matrix(cm, class_labels)
                            st.pyplot(fig)
                            
                            # Feature importance
                            if results['results']['models'][best_model]['feature_importance']:
                                st.markdown("#### 🎯 Importance des features (meilleur modèle)")
                                
                                importance = results['results']['models'][best_model]['feature_importance']
                                fig = viz.plot_feature_importance(importance, top_n=15)
                                st.pyplot(fig)
                            
                            # Code Python
                            with st.expander("💻 Code Python généré"):
                                st.code(results['python_code'], language='python')
                            
                            # Explication
                            with st.expander("📚 Interprétation"):
                                st.markdown(results['explanations']['interpretation'])
                        else:
                            st.error(f"❌ Erreur: {results.get('error', 'Inconnue')}")
            
            else:
                st.info("Type de problème non supporté pour la modélisation automatique")
    
    # ========== ONGLET 4: ÉVALUATION ==========
    with tabs[3]:
        st.header("📊 Évaluation & Diagnostics")
        st.info("Les métriques d'évaluation sont affichées dans l'onglet Modélisation après l'entraînement")
        
        if st.session_state.target:
            st.markdown(f"### Métriques recommandées pour {st.session_state.problem_type}")
            problem_type, _ = detect_problem_type(st.session_state.df[st.session_state.target])
            metrics = get_recommended_metrics(problem_type)
            
            for metric in metrics:
                st.markdown(f"- **{metric}**")
    
    # ========== ONGLET 5: SIMULATION ==========
    with tabs[4]:
        st.header("🎯 Simulation & Prédiction")
        st.info("🚧 Fonctionnalité en cours de développement")
        
        st.markdown("""
        Cette section permettra de:
        - Entrer manuellement des valeurs pour les features
        - Obtenir une prédiction du modèle entraîné
        - Visualiser les probabilités (classification)
        - Analyser la contribution de chaque feature
        
        **Important**: La variable cible ne sera JAMAIS demandée en entrée.
        """)
    
    # ========== ONGLET 6: EXPORT ==========
    with tabs[5]:
        st.header("💾 Export & Rapports")
        
        if st.session_state.df is None:
            st.warning("⚠️ Aucune donnée à exporter")
        else:
            st.subheader("Formats d'export disponibles")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📄 Données")
                
                export_format = st.selectbox("Format:", ["CSV", "Excel", "JSON"])
                
                if st.button("📥 Exporter les données"):
                    output_path = f"data/uploads/export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    if export_format == "CSV":
                        output_path += ".csv"
                        success = export_data(st.session_state.df, output_path, 'csv')
                    elif export_format == "Excel":
                        output_path += ".xlsx"
                        success = export_data(st.session_state.df, output_path, 'excel')
                    else:
                        output_path += ".json"
                        success = export_data(st.session_state.df, output_path, 'json')
                    
                    if success:
                        st.success(f"✅ Données exportées: {output_path}")
                        with open(output_path, 'rb') as f:
                            file_content = f.read()
                        st.download_button(
                            label="⬇️ Télécharger",
                            data=file_content,
                            file_name=os.path.basename(output_path)
                        )
                    else:
                        st.error("❌ Erreur lors de l'export")
            
            with col2:
                st.markdown("#### 📊 Session")
                
                if st.button("💾 Sauvegarder la session"):
                    session_data = {
                        'timestamp': datetime.now().isoformat(),
                        'target': st.session_state.target,
                        'features': st.session_state.features,
                        'problem_type': st.session_state.problem_type,
                        'data_shape': st.session_state.df.shape if st.session_state.df is not None else None,
                        'profile': st.session_state.profile
                    }
                    
                    output_path = f"data/uploads/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    success = save_session(session_data, output_path)
                    
                    if success:
                        st.success(f"✅ Session sauvegardée: {output_path}")
                    else:
                        st.error("❌ Erreur lors de la sauvegarde")
            
            # Rapport HTML
            st.markdown("---")
            st.markdown("#### 📑 Rapport HTML")
            
            if st.button("📄 Générer un rapport"):
                report_data = {
                    'target': st.session_state.target,
                    'features': st.session_state.features,
                    'problem_type': st.session_state.problem_type,
                    'data_shape': list(st.session_state.df.shape),
                    'profile': st.session_state.profile
                }
                
                html_content = generate_html_report(report_data, "Rapport DataAnalyzer 2.0")
                
                output_path = f"data/uploads/rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                st.success("✅ Rapport généré")
                st.download_button(
                    label="⬇️ Télécharger le rapport HTML",
                    data=html_content,
                    file_name="rapport_dataanalyzer.html",
                    mime="text/html"
                )

if __name__ == "__main__":
    main()
