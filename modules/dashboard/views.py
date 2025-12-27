from __future__ import annotations

from typing import Any, Dict, Optional

from pathlib import Path
import io
import json
import zipfile

import pandas as pd
from django.conf import settings
from django.http import FileResponse
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from sklearn.datasets import load_iris

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
    clustering_analysis,
)
from modules.text_analysis import analyze_text
from modules.time_series import analyze_time_series
from modules.export import (
    export_data as export_data_file,
    generate_html_report,
    generate_pdf_report,
    generate_python_code,
)
from modules.visualizations import (
    plot_distribution,
    plot_correlation_matrix,
    plot_categorical_distribution,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_residuals,
    plot_time_series,
    plot_cluster_pca_scatter,
    plot_top_terms_bar,
)
from utils.validation import (
    validate_target_not_in_features,
    validate_analysis_requirements,
)

from .models import AnalysisRun
from .forms import (
    UploadDatasetForm,
    TargetSelectionForm,
    FeatureSelectionForm,
    SamplingForm,
    CorrelationParamsForm,
    OutliersParamsForm,
    DistributionParamsForm,
    MLParamsForm,
    TextAnalysisForm,
    TimeSeriesForm,
)
from .services import (
    clear_session_state,
    get_data_context,
    set_loaded_file_path,
    save_uploaded_bytes,
    save_uploaded_file,
    set_last_results,
    get_last_results,
    export_session_payload,
    SESSION_KEY_TARGET,
    SESSION_KEY_FEATURES,
    SESSION_KEY_SAMPLING,
    SESSION_KEY_MODEL_BUNDLE_PATH,
    SESSION_KEY_MANUAL_TYPES,
)

from .ml_storage import ModelBundle, save_bundle, load_bundle


_MODEL_LABELS: dict[str, str] = {
    'logistic': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM',
    'linear': 'Linear Regression',
    'ridge': 'Ridge',
    'lasso': 'Lasso',
}


def _get_redirect_url(request: HttpRequest, wizard_step: Optional[int] = None) -> str:
    """
    Determine redirect URL based on request context.
    If request comes from wizard, redirect to appropriate wizard step.
    Otherwise, redirect to classic dashboard.
    """
    referer = request.META.get('HTTP_REFERER', '')
    
    # Check if request is from wizard - use more specific URL pattern matching
    is_wizard_context = (
        '/wizard/' in referer or 
        request.session.get('wizard_step') is not None or
        referer.endswith('/wizard/') or
        referer.endswith('/')  # Root could be wizard home
    )
    
    if is_wizard_context and '/wizard/' in referer:
        # If specific wizard step provided, use it
        if wizard_step is not None:
            return reverse('wizard_step', kwargs={'step': wizard_step})
        
        # Try to extract current step from referer using more robust parsing
        import re
        step_match = re.search(r'/wizard/step/(\d+)/', referer)
        if step_match:
            try:
                current_step = int(step_match.group(1))
                return reverse('wizard_step', kwargs={'step': current_step})
            except (ValueError, IndexError):
                pass
        
        # Default to wizard step 1 if we can't determine the step
        return reverse('wizard_step', kwargs={'step': 1})
    
    # Default to classic dashboard
    return reverse('dashboard')


def _extract_ml_summary(last_results: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not last_results or not isinstance(last_results, dict):
        return None
    if not last_results.get('success'):
        return None
    res = last_results.get('results')
    if not isinstance(res, dict):
        return None
    models = res.get('models')
    if not isinstance(models, dict) or not models:
        return None

    best = res.get('best_model')
    best_score = res.get('best_score')

    kind: str = 'unknown'
    for _, m in models.items():
        if not isinstance(m, dict):
            continue
        test_metrics = m.get('test_metrics') or {}
        if isinstance(test_metrics, dict):
            if 'r2' in test_metrics:
                kind = 'regression'
                break
            if 'f1' in test_metrics or 'roc_auc' in test_metrics:
                kind = 'classification'

    rows: list[Dict[str, Any]] = []
    for name, m in models.items():
        if not isinstance(m, dict):
            continue
        tm = m.get('train_metrics') or {}
        te = m.get('test_metrics') or {}
        rows.append({
            'name': name,
            'label': _MODEL_LABELS.get(str(name), str(name)),
            'is_best': bool(best and str(best) == str(name)),
            'train': tm if isinstance(tm, dict) else {},
            'test': te if isinstance(te, dict) else {},
        })

    rows.sort(key=lambda r: (0 if r.get('is_best') else 1, str(r.get('label') or '')))

    return {
        'kind': kind,
        'best_model': str(best) if best else None,
        'best_label': _MODEL_LABELS.get(str(best), str(best)) if best else None,
        'best_score': best_score,
        'rows': rows,
    }


def _session_key(request: HttpRequest) -> str:
    if not request.session.session_key:
        request.session.save()
    return request.session.session_key


def _store_run(request: HttpRequest, analysis_type: str, params: Dict[str, Any], results: Dict[str, Any]) -> None:
    AnalysisRun.objects.create(
        session_key=_session_key(request),
        analysis_type=analysis_type,
        params_json=params,
        summary_json={
            'success': bool(results.get('success')),
            'execution_time': results.get('execution_time'),
            'error': results.get('error'),
        },
    )


def dashboard(request: HttpRequest) -> HttpResponse:
    ctx, err = get_data_context(request)

    upload_form = UploadDatasetForm()

    columns = sorted(ctx.df.columns.tolist()) if ctx else []
    target_form = TargetSelectionForm(columns=columns, initial={'target': request.session.get(SESSION_KEY_TARGET)})

    target_selected = request.session.get(SESSION_KEY_TARGET)
    default_features = [c for c in columns if c != target_selected]
    saved_features = request.session.get(SESSION_KEY_FEATURES)
    features_initial = saved_features if saved_features is not None else default_features
    features_form = FeatureSelectionForm(columns=columns, target=target_selected, initial={'features': features_initial})

    # sampling
    sampling_initial = request.session.get(SESSION_KEY_SAMPLING) or {}
    sampling_form = SamplingForm(initial={
        'enabled': sampling_initial.get('enabled', False),
        'method': sampling_initial.get('method', 'random'),
        'n_rows': sampling_initial.get('n_rows', 10000),
        'random_state': sampling_initial.get('random_state', 42),
    })

    last_results = get_last_results(request)
    ml_summary = _extract_ml_summary(last_results)

    model_bundle_path = request.session.get(SESSION_KEY_MODEL_BUNDLE_PATH)
    model_available = bool(model_bundle_path)

    feature_schema = []
    if ctx:
        target = request.session.get(SESSION_KEY_TARGET)
        saved = request.session.get(SESSION_KEY_FEATURES)
        selected = saved if saved is not None else [c for c in ctx.df.columns.tolist() if c != target]
        for col in selected:
            info = (ctx.profile.get('columns') or {}).get(col) or {}
            feature_schema.append({'name': col, 'type': info.get('type', 'unknown')})

    columns_info = []
    manual_types = request.session.get(SESSION_KEY_MANUAL_TYPES) or {}
    if ctx:
        for col, info in (ctx.profile.get('columns') or {}).items():
            detected = info.get('type', 'unknown')
            current = manual_types.get(col) or detected
            columns_info.append({'name': col, 'detected': detected, 'current': current})

    session_key = _session_key(request)

    # historique
    runs = AnalysisRun.objects.filter(session_key=session_key).order_by('-created_at')[:20]

    warn_big = False
    if ctx and ctx.profile.get('n_rows', 0) > 10_000:
        warn_big = True

    preview_columns = []
    preview_rows = []
    numeric_columns = []
    if ctx:
        preview_columns = ctx.df.columns.tolist()
        head = ctx.df.head(10)
        preview_rows = head.astype(object).where(pd.notna(head), None).values.tolist()
        numeric_columns = ctx.df.select_dtypes(include=['number']).columns.tolist()

    # Visuels inline déjà générés (via export ZIP ou bundle)
    inline_images: list[dict[str, str]] = []
    try:
        visuals_dir = _export_dir() / f"{session_key}__visuals"
        if visuals_dir.exists() and visuals_dir.is_dir():
            for p in sorted(visuals_dir.glob('*.png')):
                inline_images.append({
                    'name': p.name,
                    'url': reverse('inline_visual', args=[p.name]),
                })
    except Exception:
        inline_images = []

    return render(
        request,
        'dashboard.html',
        {
            'data_ctx': ctx,
            'load_error': err,
            'upload_form': upload_form,
            'target_form': target_form,
            'features_form': features_form,
            'sampling_form': sampling_form,
            'last_results': last_results,
            'ml_summary': ml_summary,
            'runs': runs,
            'warn_big': warn_big,
            'preview_columns': preview_columns,
            'preview_rows': preview_rows,
            'numeric_columns': numeric_columns,
            'feature_schema': feature_schema,
            'model_available': model_available,
            'columns_info': columns_info,
            'inline_images': inline_images,
        },
    )


def _safe_name(name: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in {'_', '-', '.'} else '_' for ch in str(name))[:120]


def _save_fig_to_dir(base_dir: Path, fig, stem: str, created: list[Path]) -> None:
    stem = _safe_name(stem)
    png_path = base_dir / f"{stem}.png"
    svg_path = base_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    fig.savefig(svg_path, bbox_inches='tight')
    created.extend([png_path, svg_path])
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass


def _generate_visualizations_files(request: HttpRequest, ctx, base_dir: Path) -> list[Path]:
    """Génère les visuels (PNG + SVG) et renvoie la liste des fichiers créés."""
    session_key = _session_key(request)
    df = ctx.df
    profile_cols = (ctx.profile.get('columns') or {})
    last = get_last_results(request) or {}

    created: list[Path] = []

    # Visuels contextuels (dernier entraînement ML) si bundle présent
    try:
        bundle_path = request.session.get(SESSION_KEY_MODEL_BUNDLE_PATH)
        bundle = load_bundle(bundle_path) if bundle_path else None

        if bundle is not None and bundle.model is not None and bundle.target and bundle.features:
            run_type = 'regression' if bundle.problem_type == 'regression' else 'classification'
            last_run = (
                AnalysisRun.objects
                .filter(session_key=session_key, analysis_type=run_type)
                .order_by('-created_at')
                .first()
            )
            params = (last_run.params_json if last_run else None) or {}
            test_size = float(params.get('test_size', 0.2))
            random_state = int(params.get('random_state', 42))

            X = df[bundle.features].copy()
            y = df[bundle.target].copy()
            mask = ~y.isna()
            X = X[mask]
            y = y[mask]

            if len(X) >= 20:
                encoders = bundle.encoders or {}
                impute_values = getattr(bundle, 'impute_values', None) or {}
                missing_token = getattr(bundle, 'missing_token', '__MISSING__')
                for col in X.columns:
                    if col in encoders:
                        try:
                            X[col] = pd.Series(X[col]).astype('string').fillna(missing_token)
                            X[col] = encoders[col].transform(X[col])
                        except Exception:
                            X[col] = -1
                    else:
                        s = pd.to_numeric(X[col], errors='coerce')
                        s = s.fillna(float(impute_values.get(col, 0.0)))
                        X[col] = s

                if bundle.target_encoder is not None:
                    try:
                        y_enc = pd.Series(bundle.target_encoder.transform(y.astype(str)), index=y.index)
                    except Exception:
                        y_enc = y
                else:
                    y_enc = y

                from sklearn.model_selection import train_test_split
                from sklearn.metrics import confusion_matrix as sk_confusion_matrix

                if bundle.problem_type == 'regression':
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_enc, test_size=test_size, random_state=random_state
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
                    )

                if bundle.scaler is not None:
                    X_train = pd.DataFrame(bundle.scaler.transform(X_train), columns=X.columns, index=X_train.index)
                    X_test = pd.DataFrame(bundle.scaler.transform(X_test), columns=X.columns, index=X_test.index)

                y_pred = bundle.model.predict(X_test)

                fi = None
                try:
                    lr = last.get('results') or {}
                    best_name = lr.get('best_model')
                    models = lr.get('models') or {}
                    if best_name and isinstance(models, dict) and isinstance(models.get(best_name), dict):
                        fi = models[best_name].get('feature_importance')
                except Exception:
                    fi = None

                if not isinstance(fi, dict) or not fi:
                    fi = {}
                    if hasattr(bundle.model, 'feature_importances_'):
                        try:
                            for feat, imp in zip(X.columns, bundle.model.feature_importances_):
                                fi[str(feat)] = float(imp)
                        except Exception:
                            pass
                    elif hasattr(bundle.model, 'coef_'):
                        try:
                            import numpy as _np
                            arr = _np.abs(getattr(bundle.model, 'coef_'))
                            if getattr(arr, 'ndim', 1) > 1:
                                arr = arr.mean(axis=0)
                            for feat, coef in zip(X.columns, arr):
                                fi[str(feat)] = float(coef)
                        except Exception:
                            pass

                if fi:
                    fig = plot_feature_importance(fi, top_n=min(15, len(fi)), title='Importance des variables (modèle)')
                    _save_fig_to_dir(base_dir, fig, 'ml__feature_importance', created)

                if bundle.problem_type == 'regression':
                    try:
                        fig = plot_residuals(y_true=y_test.to_numpy(), y_pred=y_pred, title='Régression - Résidus')
                        _save_fig_to_dir(base_dir, fig, 'ml__residuals', created)
                    except Exception:
                        pass
                else:
                    try:
                        cm = sk_confusion_matrix(y_test, y_pred)
                        labels = None
                        if bundle.target_encoder is not None:
                            try:
                                labels = [str(x) for x in getattr(bundle.target_encoder, 'classes_', [])]
                            except Exception:
                                labels = None
                        if not labels:
                            labels = [str(x) for x in sorted(set(y_enc.unique().tolist()))]
                        fig = plot_confusion_matrix(cm=cm, labels=labels, title='Classification - Matrice de confusion')
                        _save_fig_to_dir(base_dir, fig, 'ml__confusion_matrix', created)
                    except Exception:
                        pass

                    try:
                        import numpy as _np
                        if hasattr(bundle.model, 'predict_proba'):
                            proba = bundle.model.predict_proba(X_test)
                            if proba is not None and getattr(proba, 'shape', (0, 0))[1] == 2:
                                from sklearn.metrics import roc_curve as sk_roc_curve, roc_auc_score as sk_roc_auc
                                fpr, tpr, _ = sk_roc_curve(y_test, proba[:, 1])
                                auc = float(sk_roc_auc(y_test, proba[:, 1]))
                                fig = plot_roc_curve(fpr=_np.asarray(fpr), tpr=_np.asarray(tpr), auc=auc, title='Classification - ROC')
                                _save_fig_to_dir(base_dir, fig, 'ml__roc_curve', created)
                    except Exception:
                        pass
    except Exception:
        pass

    # Visuels contextuels: séries temporelles
    try:
        last_ts = (
            AnalysisRun.objects
            .filter(session_key=session_key, analysis_type='time_series')
            .order_by('-created_at')
            .first()
        )
        ts_params = (last_ts.params_json if last_ts else None) or {}
        date_cols = [c for c, info in profile_cols.items() if info.get('type') == 'date']
        date_col = ts_params.get('date_column') or (date_cols[0] if date_cols else None)
        value_col = ts_params.get('value_column')
        if not value_col:
            nums = df.select_dtypes(include=['number']).columns.tolist()
            value_col = nums[0] if nums else None

        if date_col and value_col and date_col in df.columns and value_col in df.columns:
            ts_df = df[[date_col, value_col]].copy()
            ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors='coerce')
            ts_df[value_col] = pd.to_numeric(ts_df[value_col], errors='coerce')
            ts_df = ts_df.dropna().sort_values(date_col)

            if len(ts_df) > 5000:
                import numpy as _np
                idx = _np.linspace(0, len(ts_df) - 1, 5000).astype(int)
                ts_df = ts_df.iloc[idx]

            if len(ts_df) >= 10:
                fig = plot_time_series(
                    dates=ts_df[date_col],
                    values=ts_df[value_col],
                    title=f"Série temporelle - {value_col} par {date_col}",
                )
                _save_fig_to_dir(base_dir, fig, f"time_series__{value_col}__by__{date_col}", created)
    except Exception:
        pass

    # Visuels contextuels: clustering (PCA scatter)
    try:
        last_cl = (
            AnalysisRun.objects
            .filter(session_key=session_key, analysis_type='clustering')
            .order_by('-created_at')
            .first()
        )
        cl_params = (last_cl.params_json if last_cl else None) or {}
        method = (cl_params.get('method') or 'kmeans').lower().strip()
        n_clusters = int(cl_params.get('n_clusters', 3))
        scale_features = bool(cl_params.get('scale', True))
        features = cl_params.get('features')
        if not isinstance(features, list) or not features:
            features = df.select_dtypes(include=['number']).columns.tolist()[:6]
        features = [c for c in features if c in df.columns]

        if len(features) >= 2:
            X = df[features].copy()
            for c in features:
                X[c] = pd.to_numeric(X[c], errors='coerce')
            X = X.dropna()

            if len(X) >= 20:
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans, DBSCAN
                from sklearn.decomposition import PCA

                Xv = X.values
                if scale_features:
                    Xv = StandardScaler().fit_transform(Xv)

                if method == 'dbscan':
                    eps = float(cl_params.get('eps', 0.5))
                    min_samples = int(cl_params.get('min_samples', 5))
                    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xv)
                    title = f"Clustering DBSCAN (PCA) - {len(set(labels))} labels"
                else:
                    labels = KMeans(n_clusters=max(2, n_clusters), random_state=42, n_init=10).fit_predict(Xv)
                    title = f"Clustering K-Means (PCA) - k={max(2, n_clusters)}"

                comps = PCA(n_components=2, random_state=42).fit_transform(Xv)
                fig = plot_cluster_pca_scatter(comps, labels, title=title)
                _save_fig_to_dir(base_dir, fig, f"clustering__pca_scatter__{method}", created)
    except Exception:
        pass

    # Visuels contextuels: analyse texte (top termes)
    try:
        last_tx = (
            AnalysisRun.objects
            .filter(session_key=session_key, analysis_type='text_analysis')
            .order_by('-created_at')
            .first()
        )
        tx_params = (last_tx.params_json if last_tx else None) or {}
        method = (tx_params.get('method') or 'basic').lower().strip()
        top_k = int(tx_params.get('top_k', 20))

        text_col = tx_params.get('text_column')
        if not text_col:
            text_candidates = [c for c, info in profile_cols.items() if info.get('type') == 'text']
            text_col = text_candidates[0] if text_candidates else None

        if text_col and text_col in df.columns:
            res = analyze_text(df, text_column=text_col, params={'method': method, 'top_k': top_k})
            if res.get('success'):
                r = res.get('results') or {}
                if method == 'tfidf' and r.get('tfidf_top_terms'):
                    terms = [(str(t), float(s)) for t, s in r.get('tfidf_top_terms')][:top_k]
                    fig = plot_top_terms_bar(terms, title=f"Top termes TF-IDF - {text_col}")
                    _save_fig_to_dir(base_dir, fig, f"text__tfidf_top_terms__{text_col}", created)
                elif r.get('top_words'):
                    terms = [(str(t), float(c)) for t, c in r.get('top_words')][:top_k]
                    fig = plot_top_terms_bar(terms, title=f"Top mots - {text_col}")
                    _save_fig_to_dir(base_dir, fig, f"text__top_words__{text_col}", created)
    except Exception:
        pass

    # Corrélations
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig = plot_correlation_matrix(corr_matrix=corr, method='pearson', threshold=0.0)
        _save_fig_to_dir(base_dir, fig, 'correlation_matrix', created)

    # Distributions (jusqu'à 3)
    for col in numeric_cols[:3]:
        s = df[col]
        fig = plot_distribution(s, title=f"Distribution - {col}", bins=30, show_kde=True)
        _save_fig_to_dir(base_dir, fig, f"distribution__{col}", created)

    # Catégorielles (jusqu'à 2)
    cat_cols = [c for c, info in profile_cols.items() if info.get('type') in {'categorical', 'text', 'boolean'}]
    for col in cat_cols[:2]:
        s = df[col]
        fig = plot_categorical_distribution(s.astype('string'), top_n=10, title=f"Catégories - {col}")
        _save_fig_to_dir(base_dir, fig, f"categorical__{col}", created)

    return [p for p in created if p.exists()]


@require_POST
def set_manual_types(request: HttpRequest) -> HttpResponse:
    # Parse les champs type__<col>
    mapping: Dict[str, str] = {}
    for k, v in request.POST.items():
        if not k.startswith('type__'):
            continue
        col = k[len('type__'):]
        if v in {'numeric', 'categorical', 'text', 'date', 'boolean'}:
            mapping[col] = v

    request.session[SESSION_KEY_MANUAL_TYPES] = mapping
    return redirect(_get_redirect_url(request))


def _sanitize_results_for_session(results: Dict[str, Any]) -> Dict[str, Any]:
    # Copie shallow
    out: Dict[str, Any] = {k: v for k, v in results.items() if k != 'results'}
    res = results.get('results') or {}
    if isinstance(res, dict):
        cleaned = dict(res)
        # Retirer objets non sérialisables
        for k in ['scaler', 'encoders', 'target_encoder']:
            if k in cleaned:
                cleaned.pop(k)
        models = cleaned.get('models')
        if isinstance(models, dict):
            cleaned_models = {}
            for name, m in models.items():
                if not isinstance(m, dict):
                    continue
                m2 = dict(m)
                if 'model' in m2:
                    m2.pop('model')
                cleaned_models[name] = m2
            cleaned['models'] = cleaned_models
        out['results'] = cleaned
    else:
        out['results'] = res
    return out


@require_POST
def reset_session(request: HttpRequest) -> HttpResponse:
    clear_session_state(request)
    return redirect(_get_redirect_url(request))


@require_POST
def load_titanic(request: HttpRequest) -> HttpResponse:
    # Dataset d'exemple inclus dans le repo (modules/data)
    set_loaded_file_path(request, str(settings.DATA_DIR / 'Titanic-Dataset.csv'), separator=',')
    return redirect(_get_redirect_url(request, wizard_step=1))


@require_POST
def load_iris(request: HttpRequest) -> HttpResponse:
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['Species'] = df['target'].map(dict(enumerate(iris.target_names)))
    df = df.drop(columns=['target'])

    csv_bytes = df.to_csv(index=False).encode('utf-8')
    path = save_uploaded_bytes(request, 'iris.csv', csv_bytes)
    set_loaded_file_path(request, path, separator=',')
    return redirect(_get_redirect_url(request, wizard_step=1))


@require_POST
def upload_dataset(request: HttpRequest) -> HttpResponse:
    form = UploadDatasetForm(request.POST, request.FILES)
    if not form.is_valid():
        set_last_results(request, {'success': False, 'error': "Upload invalide (formulaire)."})
        return redirect(_get_redirect_url(request))

    f = form.cleaned_data['file']
    sep = form.cleaned_data['separator']

    # Formats supportés (cohérent avec modules.data_loader.load_data)
    name = getattr(f, 'name', '') or ''
    suffix = Path(name).suffix.lower()
    allowed = {'.csv', '.txt', '.xlsx', '.xls', '.json'}
    if suffix and suffix not in allowed:
        set_last_results(request, {'success': False, 'error': f"Format non supporté: {suffix}"})
        return redirect(_get_redirect_url(request))

    # Limite 100MB
    if f.size > 100 * 1024 * 1024:
        set_last_results(request, {'success': False, 'error': 'Fichier trop volumineux (limite 100MB).'})
        return redirect(_get_redirect_url(request))

    try:
        path = save_uploaded_file(request, f)
    except Exception as e:
        set_last_results(request, {'success': False, 'error': f"Échec sauvegarde upload: {str(e)}"})
        return redirect(_get_redirect_url(request))

    # Le séparateur ne s'applique qu'aux CSV/TXT
    if suffix in {'.csv', '.txt'}:
        set_loaded_file_path(request, path, separator=sep)
    else:
        set_loaded_file_path(request, path, separator=',')
    return redirect(_get_redirect_url(request, wizard_step=1))


@require_POST
def set_target_and_features(request: HttpRequest) -> HttpResponse:
    ctx, err = get_data_context(request)
    if not ctx:
        set_last_results(request, {'success': False, 'error': err or 'Aucune donnée chargée.'})
        return redirect(_get_redirect_url(request))

    columns = ctx.df.columns.tolist()
    target_form = TargetSelectionForm(request.POST, columns=columns)
    if not target_form.is_valid():
        set_last_results(request, {'success': False, 'error': 'Sélection invalide.'})
        return redirect(_get_redirect_url(request))

    target = target_form.cleaned_data['target']

    # Recréer le form features en excluant la cible des choix.
    features_form = FeatureSelectionForm(request.POST, columns=columns, target=target)
    if not features_form.is_valid():
        set_last_results(request, {'success': False, 'error': 'Sélection des features invalide.'})
        return redirect(_get_redirect_url(request))
    features = features_form.cleaned_data.get('features') or []

    is_valid, msg = validate_target_not_in_features(features, target)
    if not is_valid:
        set_last_results(request, {'success': False, 'error': msg})
        return redirect(_get_redirect_url(request))

    # Parse manual type changes from POST data (fields starting with type__)
    manual_types: Dict[str, str] = {}
    for k, v in request.POST.items():
        if not k.startswith('type__'):
            continue
        col = k[len('type__'):]
        if v in {'numeric', 'categorical', 'text', 'date', 'boolean'}:
            manual_types[col] = v

    # Save all changes in one transaction
    request.session[SESSION_KEY_TARGET] = target
    request.session[SESSION_KEY_FEATURES] = [f for f in features if f != target]
    request.session[SESSION_KEY_MANUAL_TYPES] = manual_types
    
    # Set success message
    set_last_results(request, {
        'success': True, 
        'message': 'Configuration enregistrée avec succès. Variable cible, types et features sauvegardés.'
    })

    return redirect(_get_redirect_url(request, wizard_step=4))


@require_POST
def set_sampling(request: HttpRequest) -> HttpResponse:
    form = SamplingForm(request.POST)
    if not form.is_valid():
        return redirect(_get_redirect_url(request))

    request.session[SESSION_KEY_SAMPLING] = {
        'enabled': bool(form.cleaned_data.get('enabled')),
        'method': form.cleaned_data.get('method') or 'random',
        'n_rows': form.cleaned_data.get('n_rows') or 0,
        'random_state': form.cleaned_data.get('random_state') or 42,
    }
    return redirect(_get_redirect_url(request))


def _get_features(request: HttpRequest, ctx) -> list[str]:
    cols = ctx.df.columns.tolist()
    target = request.session.get(SESSION_KEY_TARGET)
    saved = request.session.get(SESSION_KEY_FEATURES)
    if saved is not None:
        feats = [c for c in saved if c in cols and c != target]
    else:
        feats = [c for c in cols if c != target]
    return feats


@require_POST
def run_descriptive(request: HttpRequest) -> HttpResponse:
    ctx, err = get_data_context(request)
    if not ctx:
        set_last_results(request, {'success': False, 'error': err or 'Aucune donnée chargée.'})
        return redirect(_get_redirect_url(request))

    results = descriptive_statistics(ctx.df)
    _store_run(request, 'descriptive_stats', {}, results)
    set_last_results(request, results)
    return redirect(_get_redirect_url(request))


@require_POST
def run_correlation(request: HttpRequest) -> HttpResponse:
    ctx, err = get_data_context(request)
    if not ctx:
        set_last_results(request, {'success': False, 'error': err or 'Aucune donnée chargée.'})
        return redirect(_get_redirect_url(request))

    form = CorrelationParamsForm(request.POST)
    if not form.is_valid():
        set_last_results(request, {'success': False, 'error': 'Paramètres invalides.'})
        return redirect(_get_redirect_url(request))

    params = {
        'method': form.cleaned_data['method'],
        'threshold': float(form.cleaned_data['threshold']),
    }
    results = correlation_analysis(ctx.df, params=params)
    _store_run(request, 'correlation', params, results)
    set_last_results(request, results)
    return redirect(_get_redirect_url(request))


@require_POST
def run_distribution(request: HttpRequest) -> HttpResponse:
    ctx, err = get_data_context(request)
    if not ctx:
        set_last_results(request, {'success': False, 'error': err or 'Aucune donnée chargée.'})
        return redirect(_get_redirect_url(request))

    form = DistributionParamsForm(request.POST)
    if not form.is_valid():
        set_last_results(request, {'success': False, 'error': 'Paramètres invalides.'})
        return redirect(_get_redirect_url(request))

    params = {'bins': int(form.cleaned_data['bins'])}
    results = distribution_analysis(ctx.df, params=params)
    _store_run(request, 'distribution', params, results)
    set_last_results(request, results)
    return redirect(_get_redirect_url(request))


@require_POST
def run_outliers(request: HttpRequest) -> HttpResponse:
    ctx, err = get_data_context(request)
    if not ctx:
        set_last_results(request, {'success': False, 'error': err or 'Aucune donnée chargée.'})
        return redirect(_get_redirect_url(request))

    form = OutliersParamsForm(request.POST)
    if not form.is_valid():
        set_last_results(request, {'success': False, 'error': 'Paramètres invalides.'})
        return redirect(_get_redirect_url(request))

    params = {'iqr_multiplier': float(form.cleaned_data['iqr_multiplier'])}
    results = detect_outliers(ctx.df, params=params)
    _store_run(request, 'anomaly_detection', params, results)
    set_last_results(request, results)
    return redirect(_get_redirect_url(request))


@require_POST
def run_categorical(request: HttpRequest) -> HttpResponse:
    ctx, err = get_data_context(request)
    if not ctx:
        set_last_results(request, {'success': False, 'error': err or 'Aucune donnée chargée.'})
        return redirect(_get_redirect_url(request))

    results = categorical_analysis(ctx.df)
    _store_run(request, 'categorical_analysis', {}, results)
    set_last_results(request, results)
    return redirect(_get_redirect_url(request))


@require_POST
def run_stat_tests(request: HttpRequest) -> HttpResponse:
    # implémenté ensuite dans modules.eda.statistical_tests
    from modules.eda import statistical_tests

    ctx, err = get_data_context(request)
    if not ctx:
        set_last_results(request, {'success': False, 'error': err or 'Aucune donnée chargée.'})
        return redirect(_get_redirect_url(request))

    target = request.session.get(SESSION_KEY_TARGET)
    params: Dict[str, Any] = {'target': target}

    results = statistical_tests(ctx.df, target=target)
    _store_run(request, 'statistical_tests', params, results)
    set_last_results(request, results)
    return redirect(_get_redirect_url(request))


@require_POST
def run_ml_train(request: HttpRequest) -> HttpResponse:
    ctx, err = get_data_context(request)
    if not ctx:
        set_last_results(request, {'success': False, 'error': err or 'Aucune donnée chargée.'})
        return redirect(_get_redirect_url(request))

    target = request.session.get(SESSION_KEY_TARGET)
    if not target:
        set_last_results(request, {'success': False, 'error': 'Sélectionnez une variable cible.'})
        return redirect(_get_redirect_url(request))

    features = _get_features(request, ctx)

    if not features:
        set_last_results(request, {'success': False, 'error': 'Sélectionnez au moins une variable explicative (feature).'} )
        return redirect(_get_redirect_url(request))

    # Règles + prérequis
    is_valid, msg = validate_target_not_in_features(features, target)
    if not is_valid:
        set_last_results(request, {'success': False, 'error': msg})
        return redirect(_get_redirect_url(request))

    # params
    form = MLParamsForm(request.POST)
    if not form.is_valid():
        set_last_results(request, {'success': False, 'error': 'Paramètres ML invalides.'})
        return redirect(_get_redirect_url(request))

    train_size = int(form.cleaned_data['train_size']) / 100.0
    test_size = 1.0 - train_size
    params = {
        'test_size': test_size,
        'random_state': int(form.cleaned_data['random_state']),
        'scale': bool(form.cleaned_data.get('scale')),
        'models': form.cleaned_data.get('models') or [],
    }

    data_info = {
        'target_type': 'numeric' if ctx.problem_type == 'regression' else 'categorical',
        'num_features': int(ctx.df.select_dtypes(include=['number']).shape[1]),
        'cat_features': int(ctx.df.select_dtypes(include=['object', 'category']).shape[1]),
    }

    if ctx.problem_type == 'regression':
        ok, msg2 = validate_analysis_requirements('regression', data_info)
        if not ok:
            set_last_results(request, {'success': False, 'error': msg2})
            return redirect(_get_redirect_url(request))
        results = train_regression_model(ctx.df, target=target, features=features, params=params)
        _store_run(request, 'regression', params, results)
    else:
        ok, msg2 = validate_analysis_requirements('classification', data_info)
        if not ok:
            set_last_results(request, {'success': False, 'error': msg2})
            return redirect(_get_redirect_url(request))
        results = train_classification_model(ctx.df, target=target, features=features, params=params)
        _store_run(request, 'classification', params, results)

    # Persister bundle pour simulation / export
    try:
        best_name = (results.get('results') or {}).get('best_model')
        models = (results.get('results') or {}).get('models') or {}
        best_model_obj = None
        if best_name and isinstance(models, dict) and isinstance(models.get(best_name), dict):
            best_model_obj = models[best_name].get('model')
        bundle = ModelBundle(
            model=best_model_obj,
            features=features,
            target=target,
            scaler=(results.get('results') or {}).get('scaler'),
            encoders=(results.get('results') or {}).get('encoders'),
            target_encoder=(results.get('results') or {}).get('target_encoder'),
            problem_type=ctx.problem_type,
            impute_values=(results.get('results') or {}).get('impute_values'),
            missing_token=str((results.get('results') or {}).get('missing_token') or '__MISSING__'),
        )
        if bundle.model is not None:
            path = save_bundle(_session_key(request), bundle)
            request.session[SESSION_KEY_MODEL_BUNDLE_PATH] = path
    except Exception:
        # La simulation restera indisponible si la persistance échoue
        pass

    set_last_results(request, _sanitize_results_for_session(results))
    return redirect(_get_redirect_url(request))


@require_POST
def predict(request: HttpRequest) -> HttpResponse:
    ctx, err = get_data_context(request)
    if not ctx:
        set_last_results(request, {'success': False, 'error': err or 'Aucune donnée chargée.'})
        return redirect(_get_redirect_url(request))

    bundle_path = request.session.get(SESSION_KEY_MODEL_BUNDLE_PATH)
    if not bundle_path:
        set_last_results(request, {'success': False, 'error': 'Aucun modèle disponible. Lancez un entraînement ML.'})
        return redirect(_get_redirect_url(request))

    bundle = load_bundle(bundle_path)
    if bundle.model is None:
        set_last_results(request, {'success': False, 'error': 'Modèle introuvable dans le bundle.'})
        return redirect(_get_redirect_url(request))

    # Construire une ligne d'entrée à partir des features (sans jamais demander la cible)
    row: Dict[str, Any] = {}
    for feat in bundle.features:
        row[feat] = request.POST.get(f"feat__{feat}")

    X = pd.DataFrame([row])

    # Encodage + imputation robustes
    encoders = bundle.encoders or {}
    impute_values = getattr(bundle, 'impute_values', None) or {}
    missing_token = getattr(bundle, 'missing_token', '__MISSING__')

    for col in X.columns:
        if col in encoders:
            try:
                X[col] = pd.Series(X[col]).astype('string').fillna(missing_token)
                X[col] = encoders[col].transform(X[col])
            except Exception:
                # Catégorie inconnue / encodeur non compatible -> valeur sentinelle
                X[col] = -1
        else:
            s = pd.to_numeric(X[col], errors='coerce')
            s = s.fillna(float(impute_values.get(col, 0.0)))
            X[col] = s

    if bundle.scaler is not None:
        X_scaled = bundle.scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)

    pred = bundle.model.predict(X)

    payload: Dict[str, Any] = {
        'success': True,
        'results': {
            'target': bundle.target,
            'prediction_raw': pred.tolist(),
        },
        'explanations': {
            'method': 'Prédiction sur une observation (simulation)',
            'interpretation': "La prédiction est produite à partir des features sélectionnées, sans jamais demander la cible.",
            'warnings': [
                'Les valeurs inédites (catégories non vues) peuvent être refusées.',
                'Les valeurs manquantes peuvent dégrader la qualité.'
            ]
        }
    }

    # Décodage classification si encoder présent
    if bundle.target_encoder is not None:
        try:
            decoded = bundle.target_encoder.inverse_transform(pred)
            payload['results']['prediction'] = [str(x) for x in decoded]
        except Exception:
            pass

    # Probabilités si dispo
    if hasattr(bundle.model, 'predict_proba'):
        try:
            proba = bundle.model.predict_proba(X)
            payload['results']['prediction_proba'] = proba.tolist()
        except Exception:
            pass

    set_last_results(request, payload)
    return redirect(_get_redirect_url(request))


@require_POST
def run_clustering(request: HttpRequest) -> HttpResponse:
    ctx, err = get_data_context(request)
    if not ctx:
        set_last_results(request, {'success': False, 'error': err or 'Aucune donnée chargée.'})
        return redirect(_get_redirect_url(request))

    numeric_cols = ctx.df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) < 2:
        set_last_results(request, {'success': False, 'error': 'Clustering nécessite au moins 2 variables numériques.'})
        return redirect(_get_redirect_url(request))

    features = numeric_cols[:6]
    params: Dict[str, Any] = {'method': 'kmeans', 'n_clusters': 3, 'scale': True, 'features': features}
    results = clustering_analysis(ctx.df, features=features, params=params)
    _store_run(request, 'clustering', params, results)
    set_last_results(request, results)
    return redirect(_get_redirect_url(request))


@require_POST
def run_text(request: HttpRequest) -> HttpResponse:
    ctx, err = get_data_context(request)
    if not ctx:
        set_last_results(request, {'success': False, 'error': err or 'Aucune donnée chargée.'})
        return redirect(_get_redirect_url(request))

    text_cols = [c for c, info in ctx.profile.get('columns', {}).items() if info.get('type') == 'text']
    if not text_cols:
        set_last_results(request, {'success': False, 'error': 'Aucune colonne texte détectée.'})
        return redirect(_get_redirect_url(request))

    form = TextAnalysisForm(request.POST, columns=text_cols)
    if not form.is_valid():
        set_last_results(request, {'success': False, 'error': 'Paramètres invalides.'})
        return redirect(_get_redirect_url(request))

    params = {
        'method': form.cleaned_data['method'],
        'top_k': int(form.cleaned_data['top_k']),
        'text_column': form.cleaned_data['text_column'],
        'ngram_max': int(form.cleaned_data.get('ngram_max') or 1),
        'max_features': int(form.cleaned_data.get('max_features') or 2000),
        'stop_words': (form.cleaned_data.get('stop_words') or None),
        'compute_similarity': True,
        'similarity_top_n': int(form.cleaned_data.get('similarity_top_n') or 10),
    }
    results = analyze_text(ctx.df, text_column=form.cleaned_data['text_column'], params=params)
    _store_run(request, 'text_analysis', params, results)
    set_last_results(request, results)
    return redirect(_get_redirect_url(request))


@require_POST
def run_time_series(request: HttpRequest) -> HttpResponse:
    ctx, err = get_data_context(request)
    if not ctx:
        set_last_results(request, {'success': False, 'error': err or 'Aucune donnée chargée.'})
        return redirect(_get_redirect_url(request))

    date_cols = [c for c, info in ctx.profile.get('columns', {}).items() if info.get('type') == 'date']
    value_cols = ctx.df.select_dtypes(include=['number']).columns.tolist()
    if not date_cols or not value_cols:
        set_last_results(request, {'success': False, 'error': 'Séries temporelles nécessite une colonne date et une colonne numérique.'})
        return redirect(_get_redirect_url(request))

    form = TimeSeriesForm(request.POST, date_columns=date_cols, value_columns=value_cols)
    if not form.is_valid():
        set_last_results(request, {'success': False, 'error': 'Paramètres invalides.'})
        return redirect(_get_redirect_url(request))

    params: Dict[str, Any] = {
        'date_column': form.cleaned_data['date_column'],
        'value_column': form.cleaned_data['value_column'],
    }
    results = analyze_time_series(
        ctx.df,
        date_column=form.cleaned_data['date_column'],
        value_column=form.cleaned_data['value_column'],
        params=params,
    )
    _store_run(request, 'time_series', params, results)
    set_last_results(request, results)
    return redirect(_get_redirect_url(request))


def export_session_json(request: HttpRequest) -> HttpResponse:
    ctx, _ = get_data_context(request)
    payload = export_session_payload(request, ctx)
    return JsonResponse(payload)


def _export_dir() -> Path:
    p = Path(settings.EXPORT_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def export_report_html(request: HttpRequest) -> HttpResponse:
    ctx, err = get_data_context(request)
    if not ctx:
        set_last_results(request, {'success': False, 'error': err or 'Aucune donnée chargée.'})
        return redirect(_get_redirect_url(request))

    payload = export_session_payload(request, ctx)
    html = generate_html_report(payload, title="DataAnalyzer V2 - Rapport")
    out = _export_dir() / f"{_session_key(request)}__report.html"
    out.write_text(html, encoding='utf-8')
    return FileResponse(open(out, 'rb'), as_attachment=True, filename=out.name, content_type='text/html; charset=utf-8')


def export_report_pdf(request: HttpRequest) -> HttpResponse:
    ctx, err = get_data_context(request)
    if not ctx:
        set_last_results(request, {'success': False, 'error': err or 'Aucune donnée chargée.'})
        return redirect(_get_redirect_url(request))

    payload = export_session_payload(request, ctx)
    out = _export_dir() / f"{_session_key(request)}__report.pdf"
    ok = generate_pdf_report(payload, filepath=str(out), title="DataAnalyzer V2 - Rapport")
    if not ok:
        set_last_results(request, {'success': False, 'error': 'Échec génération PDF.'})
        return redirect(_get_redirect_url(request))
    return FileResponse(open(out, 'rb'), as_attachment=True, filename=out.name, content_type='application/pdf')


def export_data(request: HttpRequest) -> HttpResponse:
    ctx, err = get_data_context(request)
    if not ctx:
        set_last_results(request, {'success': False, 'error': err or 'Aucune donnée chargée.'})
        return redirect(_get_redirect_url(request))

    fmt = (request.GET.get('format') or 'csv').lower().strip()
    if fmt not in {'csv', 'excel', 'json'}:
        fmt = 'csv'

    suffix = 'csv' if fmt == 'csv' else ('xlsx' if fmt == 'excel' else 'json')
    out = _export_dir() / f"{_session_key(request)}__data.{suffix}"
    ok = export_data_file(ctx.df, str(out), format=fmt)
    if not ok:
        set_last_results(request, {'success': False, 'error': 'Échec export données.'})
        return redirect(_get_redirect_url(request))

    ctype = 'text/csv' if fmt == 'csv' else ('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' if fmt == 'excel' else 'application/json')
    return FileResponse(open(out, 'rb'), as_attachment=True, filename=out.name, content_type=ctype)


def export_model_bundle(request: HttpRequest) -> HttpResponse:
    bundle_path = request.session.get(SESSION_KEY_MODEL_BUNDLE_PATH)
    if not bundle_path:
        set_last_results(request, {'success': False, 'error': 'Aucun modèle disponible. Entraînez un modèle avant export.'})
        return redirect(_get_redirect_url(request))

    p = Path(bundle_path)
    if not p.exists():
        set_last_results(request, {'success': False, 'error': 'Le fichier modèle est introuvable (bundle manquant).'})
        return redirect(_get_redirect_url(request))

    return FileResponse(open(p, 'rb'), as_attachment=True, filename=p.name, content_type='application/octet-stream')


def export_python_code_last(request: HttpRequest) -> HttpResponse:
    last = get_last_results(request) or {}
    code = last.get('python_code')
    if not code:
        # Fallback minimal
        code = generate_python_code(analysis_type=str(last.get('analysis_type') or 'analysis'), params=last.get('params') or {})

    out = _export_dir() / f"{_session_key(request)}__repro.py"
    out.write_text(str(code), encoding='utf-8')
    return FileResponse(open(out, 'rb'), as_attachment=True, filename=out.name, content_type='text/x-python; charset=utf-8')


def export_visualizations_zip(request: HttpRequest) -> HttpResponse:
    """Exporte un ZIP de visualisations (PNG + SVG).

    Génère des graphiques génériques basés sur le dataset courant:
    - corrélations (si >=2 colonnes numériques)
    - distributions (jusqu'à 3 colonnes numériques)
    - distribution catégorielle (jusqu'à 2 colonnes catégorielles)
    """
    ctx, err = get_data_context(request)
    if not ctx:
        set_last_results(request, {'success': False, 'error': err or 'Aucune donnée chargée.'})
        return redirect(_get_redirect_url(request))

    session_key = _session_key(request)
    base_dir = _export_dir() / f"{session_key}__visuals"
    base_dir.mkdir(parents=True, exist_ok=True)
    created = _generate_visualizations_files(request, ctx, base_dir)

    if not created:
        set_last_results(request, {'success': False, 'error': 'Aucune visualisation générable (dataset trop petit ou types non compatibles).'} )
        return redirect(_get_redirect_url(request))

    zip_path = _export_dir() / f"{session_key}__visuals.zip"
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for p in created:
            if p.exists():
                zf.write(p, arcname=p.name)

    return FileResponse(open(zip_path, 'rb'), as_attachment=True, filename=zip_path.name, content_type='application/zip')


def generate_inline_visuals(request: HttpRequest) -> HttpResponse:
    """Génère les visuels dans le dossier de session et revient au dashboard."""
    ctx, err = get_data_context(request)
    if not ctx:
        set_last_results(request, {'success': False, 'error': err or 'Aucune donnée chargée.'})
        return redirect(_get_redirect_url(request))

    session_key = _session_key(request)
    base_dir = _export_dir() / f"{session_key}__visuals"
    base_dir.mkdir(parents=True, exist_ok=True)
    created = _generate_visualizations_files(request, ctx, base_dir)
    if not created:
        set_last_results(request, {'success': False, 'error': 'Aucun visuel généré.'})
    return redirect(_get_redirect_url(request))


def inline_visual(request: HttpRequest, filename: str) -> HttpResponse:
    """Sert un PNG généré pour la session courante."""
    session_key = _session_key(request)
    fname = _safe_name(filename)
    if fname != filename or '/' in filename or '\\' in filename:
        return HttpResponse(status=400)
    if not (fname.endswith('.png') or fname.endswith('.svg')):
        return HttpResponse(status=400)

    p = _export_dir() / f"{session_key}__visuals" / fname
    if not p.exists() or not p.is_file():
        return HttpResponse(status=404)

    ctype = 'image/png' if p.suffix.lower() == '.png' else 'image/svg+xml'
    return FileResponse(open(p, 'rb'), content_type=ctype)


def export_bundle_zip(request: HttpRequest) -> HttpResponse:
    """Exporte un ZIP unique contenant session + rapports + données + code + modèle + visuels."""
    ctx, err = get_data_context(request)
    if not ctx:
        set_last_results(request, {'success': False, 'error': err or 'Aucune donnée chargée.'})
        return redirect(_get_redirect_url(request))

    session_key = _session_key(request)
    out_zip = _export_dir() / f"{session_key}__bundle.zip"

    warnings: list[str] = []

    payload = export_session_payload(request, ctx)
    try:
        runs = list(
            AnalysisRun.objects
            .filter(session_key=session_key)
            .order_by('-created_at')[:200]
            .values('created_at', 'analysis_type', 'params_json', 'summary_json')
        )
        for r in runs:
            try:
                r['created_at'] = r['created_at'].isoformat() if r.get('created_at') else None
            except Exception:
                pass
    except Exception:
        runs = []
        warnings.append('Impossible de récupérer l\'historique (AnalysisRun).')

    # Préparer artefacts sur disque (évite de tout garder en RAM)
    report_html_path = _export_dir() / f"{session_key}__bundle_report.html"
    report_pdf_path = _export_dir() / f"{session_key}__bundle_report.pdf"
    data_csv_path = _export_dir() / f"{session_key}__bundle_data.csv"
    data_json_path = _export_dir() / f"{session_key}__bundle_data.json"
    repro_py_path = _export_dir() / f"{session_key}__bundle_repro.py"

    try:
        html = generate_html_report(payload, title="DataAnalyzer V2 - Rapport")
        report_html_path.write_text(html, encoding='utf-8')
    except Exception:
        warnings.append('Échec génération rapport HTML.')

    try:
        ok = generate_pdf_report(payload, filepath=str(report_pdf_path), title="DataAnalyzer V2 - Rapport")
        if not ok:
            warnings.append('Échec génération rapport PDF.')
    except Exception:
        warnings.append('Échec génération rapport PDF.')

    try:
        export_data_file(ctx.df, str(data_csv_path), format='csv')
    except Exception:
        warnings.append('Échec export données CSV.')

    try:
        export_data_file(ctx.df, str(data_json_path), format='json')
    except Exception:
        warnings.append('Échec export données JSON.')

    try:
        last = get_last_results(request) or {}
        code = last.get('python_code')
        if not code:
            code = generate_python_code(analysis_type=str(last.get('analysis_type') or 'analysis'), params=last.get('params') or {})
        repro_py_path.write_text(str(code), encoding='utf-8')
    except Exception:
        warnings.append('Échec génération code Python reproductible.')

    # Visuels
    visuals_dir = _export_dir() / f"{session_key}__visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    try:
        _generate_visualizations_files(request, ctx, visuals_dir)
    except Exception:
        warnings.append('Échec génération visuels.')

    bundle_path = request.session.get(SESSION_KEY_MODEL_BUNDLE_PATH)
    model_path = Path(bundle_path) if bundle_path else None

    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        # JSON session + historique
        zf.writestr('meta/session.json', json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        zf.writestr('meta/runs.json', json.dumps(runs, ensure_ascii=False, indent=2, default=str))
        zf.writestr('meta/last_results.json', json.dumps(get_last_results(request) or {}, ensure_ascii=False, indent=2, default=str))

        if warnings:
            zf.writestr('meta/warnings.txt', '\n'.join(warnings))

        # Rapports
        if report_html_path.exists():
            zf.write(report_html_path, arcname='report/report.html')
        if report_pdf_path.exists():
            zf.write(report_pdf_path, arcname='report/report.pdf')

        # Données
        if data_csv_path.exists():
            zf.write(data_csv_path, arcname='data/data.csv')
        if data_json_path.exists():
            zf.write(data_json_path, arcname='data/data.json')

        # Code
        if repro_py_path.exists():
            zf.write(repro_py_path, arcname='code/repro.py')

        # Modèle
        if model_path is not None and model_path.exists() and model_path.is_file():
            zf.write(model_path, arcname=f"model/{model_path.name}")

        # Visuels
        try:
            for p in sorted(visuals_dir.glob('*.png')):
                zf.write(p, arcname=f"visuals/{p.name}")
            for p in sorted(visuals_dir.glob('*.svg')):
                zf.write(p, arcname=f"visuals/{p.name}")
        except Exception:
            pass

    return FileResponse(open(out_zip, 'rb'), as_attachment=True, filename=out_zip.name, content_type='application/zip')
