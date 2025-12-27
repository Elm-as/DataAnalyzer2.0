"""
Wizard views for step-by-step data analysis workflow
"""
from __future__ import annotations

from typing import Any, Dict, Optional, List
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from modules.data_profiler import profile_dataframe, get_data_quality_score
from modules.eda import (
    descriptive_statistics,
    correlation_analysis,
    distribution_analysis,
    detect_outliers,
    categorical_analysis,
)
from modules.ml_models import (
    train_regression_model,
    train_classification_model,
)
from utils.validation import (
    validate_target_not_in_features,
    detect_problem_type,
)

from .services import (
    clear_session_state,
    get_data_context,
    set_loaded_file_path,
    save_uploaded_file,
    set_last_results,
    get_last_results,
    SESSION_KEY_TARGET,
    SESSION_KEY_FEATURES,
)
from .forms import UploadDatasetForm

# Wizard session keys
SESSION_KEY_WIZARD_STEP = 'wizard_step'
SESSION_KEY_WIZARD_COMPLETED_STEPS = 'wizard_completed_steps'
SESSION_KEY_SELECTED_ANALYSES = 'wizard_selected_analyses'
SESSION_KEY_QUALITY_WARNINGS = 'wizard_quality_warnings'
SESSION_KEY_QUALITY_SUGGESTIONS = 'wizard_quality_suggestions'


def _get_wizard_step(request: HttpRequest) -> int:
    """Get current wizard step (0-7)"""
    return request.session.get(SESSION_KEY_WIZARD_STEP, 0)


def _set_wizard_step(request: HttpRequest, step: int) -> None:
    """Set current wizard step"""
    request.session[SESSION_KEY_WIZARD_STEP] = step
    
    # Mark as completed
    completed = set(request.session.get(SESSION_KEY_WIZARD_COMPLETED_STEPS, []))
    if step > 0:
        completed.add(step - 1)
    request.session[SESSION_KEY_WIZARD_COMPLETED_STEPS] = list(completed)


def _can_access_step(request: HttpRequest, step: int) -> bool:
    """Check if user can access a specific step"""
    if step == 0:
        return True
    
    ctx, _ = get_data_context(request)
    
    # Step 1: Can always access after step 0
    if step == 1:
        return True
    
    # Step 2: Need data loaded
    if step == 2:
        return ctx is not None
    
    # Step 3: Need data loaded
    if step == 3:
        return ctx is not None
    
    # Step 4: Need data loaded
    if step == 4:
        return ctx is not None
    
    # Step 5: Need target selected
    if step == 5:
        target = request.session.get(SESSION_KEY_TARGET)
        return ctx is not None and target is not None
    
    # Step 6: Need analyses selected
    if step == 6:
        analyses = request.session.get(SESSION_KEY_SELECTED_ANALYSES, [])
        return ctx is not None and len(analyses) > 0
    
    # Step 7: Need ML model trained
    if step == 7:
        from .services import SESSION_KEY_MODEL_BUNDLE_PATH
        bundle_path = request.session.get(SESSION_KEY_MODEL_BUNDLE_PATH)
        return bundle_path is not None
    
    return False


def wizard_home(request: HttpRequest) -> HttpResponse:
    """Wizard Step 0: Welcome page"""
    current_step = _get_wizard_step(request)
    
    return render(request, 'wizard/step0_welcome.html', {
        'current_step': current_step,
    })


@require_POST
def wizard_start(request: HttpRequest) -> HttpResponse:
    """Start the wizard"""
    clear_session_state(request)
    _set_wizard_step(request, 1)
    return redirect('wizard_step', step=1)


def wizard_step(request: HttpRequest, step: int) -> HttpResponse:
    """Display a wizard step"""
    
    # Check access
    if not _can_access_step(request, step):
        # Redirect to furthest accessible step
        for i in range(step - 1, -1, -1):
            if _can_access_step(request, i):
                return redirect('wizard_step', step=i)
        return redirect('wizard_home')
    
    _set_wizard_step(request, step)
    
    # Route to appropriate step handler
    if step == 1:
        return _wizard_step1_import(request)
    elif step == 2:
        return _wizard_step2_preview(request)
    elif step == 3:
        return _wizard_step3_quality(request)
    elif step == 4:
        return _wizard_step4_configuration(request)
    elif step == 5:
        return _wizard_step5_analysis_selection(request)
    elif step == 6:
        return _wizard_step6_results(request)
    elif step == 7:
        return _wizard_step7_simulation(request)
    else:
        return redirect('wizard_home')


def _wizard_step1_import(request: HttpRequest) -> HttpResponse:
    """Step 1: Import data"""
    ctx, err = get_data_context(request)
    upload_form = UploadDatasetForm()
    
    return render(request, 'wizard/step1_import.html', {
        'current_step': 1,
        'data_ctx': ctx,
        'load_error': err,
        'upload_form': upload_form,
    })


def _wizard_step2_preview(request: HttpRequest) -> HttpResponse:
    """Step 2: Preview data"""
    ctx, err = get_data_context(request)
    
    if not ctx:
        return redirect('wizard_step', step=1)
    
    # Get preview data
    preview_columns = ctx.df.columns.tolist()
    head = ctx.df.head(20)
    preview_rows = head.astype(object).where(pd.notna(head), None).values.tolist()
    
    return render(request, 'wizard/step2_preview.html', {
        'current_step': 2,
        'data_ctx': ctx,
        'preview_columns': preview_columns,
        'preview_rows': preview_rows,
    })


def _wizard_step3_quality(request: HttpRequest) -> HttpResponse:
    """Step 3: Data quality report"""
    ctx, err = get_data_context(request)
    
    if not ctx:
        return redirect('wizard_step', step=1)
    
    profile = ctx.profile
    df = ctx.df
    
    # Generate warnings and suggestions
    warnings = []
    suggestions = []
    
    # Check for duplicates
    n_duplicates = profile['quality_metrics']['n_duplicates']
    if n_duplicates > 0:
        warnings.append(f"⚠️ {n_duplicates} lignes dupliquées détectées")
    
    # Check for problematic columns
    problematic_cols = []
    for col, info in profile['columns'].items():
        pct_missing = info['pct_missing']
        if pct_missing > 50:
            problematic_cols.append(col)
        elif pct_missing > 30:
            if info['type'] not in ['numeric', 'date']:
                problematic_cols.append(col)
    
    if problematic_cols:
        warnings.append(f"⚠️ {len(problematic_cols)} colonnes problématiques : {', '.join(problematic_cols[:5])}")
        suggestions.append(f"Envisager de supprimer ou nettoyer ces colonnes : {', '.join(problematic_cols)}")
    
    # Overall quality assessment
    score, level = get_data_quality_score(profile)
    if score >= 70:
        suggestions.append("✅ Qualité des données suffisante pour l'analyse")
    elif score >= 50:
        suggestions.append("⚠️ Qualité des données moyenne. Nettoyage recommandé avant analyse.")
    else:
        suggestions.append("❌ Qualité des données faible. Nettoyage nécessaire.")
    
    # Store in session for later use
    request.session[SESSION_KEY_QUALITY_WARNINGS] = warnings
    request.session[SESSION_KEY_QUALITY_SUGGESTIONS] = suggestions
    
    return render(request, 'wizard/step3_quality.html', {
        'current_step': 3,
        'data_ctx': ctx,
        'warnings': warnings,
        'suggestions': suggestions,
        'quality_score': score,
        'quality_level': level,
    })


def _wizard_step4_configuration(request: HttpRequest) -> HttpResponse:
    """Step 4: Configuration (target, types, features)"""
    ctx, err = get_data_context(request)
    
    if not ctx:
        return redirect('wizard_step', step=1)
    
    from .forms import TargetSelectionForm, FeatureSelectionForm
    
    columns = sorted(ctx.df.columns.tolist())
    target = request.session.get(SESSION_KEY_TARGET)
    
    target_form = TargetSelectionForm(columns=columns, initial={'target': target})
    
    # Features form
    default_features = [c for c in columns if c != target]
    saved_features = request.session.get(SESSION_KEY_FEATURES)
    features_initial = saved_features if saved_features is not None else default_features
    features_form = FeatureSelectionForm(columns=columns, target=target, initial={'features': features_initial})
    
    # Column types
    from .services import SESSION_KEY_MANUAL_TYPES
    manual_types = request.session.get(SESSION_KEY_MANUAL_TYPES) or {}
    columns_info = []
    for col, info in (ctx.profile.get('columns') or {}).items():
        detected = info.get('type', 'unknown')
        current = manual_types.get(col) or detected
        columns_info.append({'name': col, 'detected': detected, 'current': current})
    
    # Get last results for success/error messages
    last_results = get_last_results(request)
    
    return render(request, 'wizard/step4_configuration.html', {
        'current_step': 4,
        'data_ctx': ctx,
        'target_form': target_form,
        'features_form': features_form,
        'columns_info': columns_info,
        'target': target,
        'last_results': last_results,
    })


def _wizard_step5_analysis_selection(request: HttpRequest) -> HttpResponse:
    """Step 5: Select analyses to perform"""
    ctx, err = get_data_context(request)
    
    if not ctx:
        return redirect('wizard_step', step=1)
    
    target = request.session.get(SESSION_KEY_TARGET)
    features = request.session.get(SESSION_KEY_FEATURES, [])
    
    if not target:
        return redirect('wizard_step', step=4)
    
    # Determine available analyses based on data characteristics
    df = ctx.df
    n_rows = len(df)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Check for date columns
    date_cols = [c for c, info in ctx.profile.get('columns', {}).items() if info.get('type') == 'date']
    has_date_column = len(date_cols) > 0
    
    # Detect problem type
    problem_type, description = detect_problem_type(df[target])
    
    # Define available analyses
    basic_analyses = [
        {
            'id': 'descriptive',
            'name': 'Statistiques descriptives',
            'duration': '~2s',
            'description': 'Moyenne, médiane, écart-type, quartiles pour chaque colonne numérique',
            'enabled': len(numeric_cols) > 0,
            'default': True,
        },
        {
            'id': 'correlation',
            'name': 'Corrélations',
            'duration': '~3s',
            'description': 'Matrice de corrélation entre les variables numériques',
            'enabled': len(numeric_cols) >= 2,
            'default': True,
        },
        {
            'id': 'distribution',
            'name': 'Distributions',
            'duration': '~4s',
            'description': 'Histogrammes et analyse de la distribution des données',
            'enabled': len(numeric_cols) > 0,
            'default': True,
        },
        {
            'id': 'outliers',
            'name': 'Détection d\'anomalies',
            'duration': '~3s',
            'description': 'Identification des valeurs aberrantes par méthode IQR',
            'enabled': len(numeric_cols) > 0,
            'default': True,
        },
        {
            'id': 'categorical',
            'name': 'Analyse catégorielle',
            'duration': '~3s',
            'description': 'Fréquences, modes et distribution des variables catégorielles',
            'enabled': len(categorical_cols) > 0,
            'default': len(categorical_cols) > 0,
        },
    ]
    
    advanced_analyses = [
        {
            'id': 'regression',
            'name': 'Régression (ML)',
            'duration': '~10s',
            'description': 'Modèles de régression : Linéaire, Ridge, Lasso, Random Forest, XGBoost',
            'enabled': problem_type == 'regression' and len(features) > 0,
            'default': problem_type == 'regression',
        },
        {
            'id': 'classification',
            'name': 'Classification (ML)',
            'duration': '~12s',
            'description': 'Logistic Regression, Random Forest, XGBoost, LightGBM',
            'enabled': problem_type in ['binary_classification', 'multiclass_classification'] and len(features) > 0,
            'default': problem_type in ['binary_classification', 'multiclass_classification'],
        },
        {
            'id': 'clustering',
            'name': 'Clustering Avancé',
            'duration': '~12s',
            'description': 'K-Means, DBSCAN avec optimisation automatique',
            'enabled': len(numeric_cols) >= 2,
            'default': False,
        },
        {
            'id': 'time_series',
            'name': 'Séries Temporelles',
            'duration': '~20s',
            'description': 'Analyse temporelle et prévisions',
            'enabled': has_date_column and len(numeric_cols) > 0,
            'default': False,
        },
    ]
    
    # Get currently selected analyses
    selected = request.session.get(SESSION_KEY_SELECTED_ANALYSES, [])
    if not selected:
        # Default selection
        selected = [a['id'] for a in basic_analyses + advanced_analyses if a.get('default', False) and a.get('enabled', False)]
        request.session[SESSION_KEY_SELECTED_ANALYSES] = selected
    
    # Calculate estimated time
    total_time = 0
    for analysis in basic_analyses + advanced_analyses:
        if analysis['id'] in selected:
            duration_str = analysis.get('duration', '~0s')
            time_val = int(duration_str.replace('~', '').replace('s', ''))
            total_time += time_val
    
    return render(request, 'wizard/step5_analysis_selection.html', {
        'current_step': 5,
        'data_ctx': ctx,
        'n_rows': n_rows,
        'n_numeric_cols': len([f for f in features if f in numeric_cols]),
        'n_selected_features': len(features),
        'basic_analyses': basic_analyses,
        'advanced_analyses': advanced_analyses,
        'selected_analyses': selected,
        'estimated_time': total_time,
        'problem_type': description,
    })


def _wizard_step6_results(request: HttpRequest) -> HttpResponse:
    """Step 6: Display results by tabs"""
    ctx, err = get_data_context(request)
    
    if not ctx:
        return redirect('wizard_step', step=1)
    
    # Get all analysis results from session
    last_results = get_last_results(request)
    
    # Get selected analyses
    selected_analyses = request.session.get(SESSION_KEY_SELECTED_ANALYSES, [])
    
    return render(request, 'wizard/step6_results.html', {
        'current_step': 6,
        'data_ctx': ctx,
        'last_results': last_results,
        'selected_analyses': selected_analyses,
    })


def _wizard_step7_simulation(request: HttpRequest) -> HttpResponse:
    """Step 7: Simulation and prediction"""
    ctx, err = get_data_context(request)
    
    if not ctx:
        return redirect('wizard_step', step=1)
    
    from .services import SESSION_KEY_MODEL_BUNDLE_PATH
    from .ml_storage import load_bundle
    
    bundle_path = request.session.get(SESSION_KEY_MODEL_BUNDLE_PATH)
    bundle = load_bundle(bundle_path) if bundle_path else None
    
    feature_schema = []
    if bundle and bundle.features:
        for col in bundle.features:
            info = (ctx.profile.get('columns') or {}).get(col) or {}
            feature_schema.append({'name': col, 'type': info.get('type', 'unknown')})
    
    return render(request, 'wizard/step7_simulation.html', {
        'current_step': 7,
        'data_ctx': ctx,
        'model_available': bundle is not None,
        'feature_schema': feature_schema,
        'bundle': bundle,
    })


@require_POST
def wizard_select_analyses(request: HttpRequest) -> HttpResponse:
    """Save selected analyses"""
    selected = request.POST.getlist('analyses')
    request.session[SESSION_KEY_SELECTED_ANALYSES] = selected
    return redirect('wizard_step', step=5)


@require_POST
def wizard_run_analyses(request: HttpRequest) -> HttpResponse:
    """Run all selected analyses"""
    ctx, err = get_data_context(request)
    if not ctx:
        set_last_results(request, {'success': False, 'error': 'Aucune donnée chargée.'})
        return redirect('wizard_step', step=1)
    
    selected_analyses = request.session.get(SESSION_KEY_SELECTED_ANALYSES, [])
    target = request.session.get(SESSION_KEY_TARGET)
    features = request.session.get(SESSION_KEY_FEATURES, [])
    
    results = {}
    
    # Run each selected analysis
    for analysis_id in selected_analyses:
        if analysis_id == 'descriptive':
            results['descriptive'] = descriptive_statistics(ctx.df)
        elif analysis_id == 'correlation':
            results['correlation'] = correlation_analysis(ctx.df, params={'method': 'pearson', 'threshold': 0.3})
        elif analysis_id == 'distribution':
            results['distribution'] = distribution_analysis(ctx.df, params={'bins': 30})
        elif analysis_id == 'outliers':
            results['outliers'] = detect_outliers(ctx.df, params={'iqr_multiplier': 1.5})
        elif analysis_id == 'categorical':
            results['categorical'] = categorical_analysis(ctx.df)
        elif analysis_id == 'regression' and target:
            params = {
                'test_size': 0.2,
                'random_state': 42,
                'scale': True,
                'models': ['linear', 'random_forest', 'xgboost']
            }
            results['regression'] = train_regression_model(ctx.df, target, features, params)
        elif analysis_id == 'classification' and target:
            params = {
                'test_size': 0.2,
                'random_state': 42,
                'scale': True,
                'models': ['logistic', 'random_forest', 'xgboost']
            }
            results['classification'] = train_classification_model(ctx.df, target, features, params)
    
    # Store results
    request.session['wizard_analysis_results'] = results
    set_last_results(request, {'success': True, 'results': results})
    
    return redirect('wizard_step', step=6)


def wizard_correlation_management(request: HttpRequest) -> HttpResponse:
    """View highly correlated features and select which to remove"""
    ctx, err = get_data_context(request)
    
    if not ctx:
        return redirect('wizard_step', step=1)
    
    features = request.session.get(SESSION_KEY_FEATURES, [])
    if not features or len(features) < 2:
        return redirect('wizard_step', step=5)
    
    # Calculate correlations
    df = ctx.df
    numeric_features = [f for f in features if f in df.select_dtypes(include=['number']).columns]
    
    if len(numeric_features) < 2:
        return redirect('wizard_step', step=5)
    
    # Compute correlation matrix
    corr_matrix = df[numeric_features].corr()
    
    # Find high correlations (> 0.7)
    threshold = 0.7
    high_correlations = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value > threshold:
                high_correlations.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'value': corr_value
                })
    
    # Sort by correlation value (highest first)
    high_correlations.sort(key=lambda x: x['value'], reverse=True)
    
    return render(request, 'wizard/step5b_correlations.html', {
        'current_step': 5,
        'data_ctx': ctx,
        'high_correlations': high_correlations,
        'threshold': threshold,
        'correlation_matrix': corr_matrix.to_dict() if len(high_correlations) > 0 else None,
    })


@require_POST
def wizard_manage_correlations(request: HttpRequest) -> HttpResponse:
    """Apply correlation management - remove selected features"""
    features_to_remove = request.POST.getlist('remove_features')
    
    current_features = request.session.get(SESSION_KEY_FEATURES, [])
    updated_features = [f for f in current_features if f not in features_to_remove]
    
    request.session[SESSION_KEY_FEATURES] = updated_features
    
    # Show success message
    if features_to_remove:
        set_last_results(request, {
            'success': True,
            'message': f"{len(features_to_remove)} feature(s) supprimée(s) pour réduire la corrélation."
        })
    
    return redirect('wizard_step', step=5)
