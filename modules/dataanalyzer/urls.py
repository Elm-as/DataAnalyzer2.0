from django.contrib import admin
from django.urls import path

from modules.dashboard import views
from modules.dashboard import wizard_views

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Wizard interface (new default)
    path('', wizard_views.wizard_home, name='wizard_home'),
    path('wizard/start/', wizard_views.wizard_start, name='wizard_start'),
    path('wizard/step/<int:step>/', wizard_views.wizard_step, name='wizard_step'),
    path('wizard/correlations/', wizard_views.wizard_correlation_management, name='wizard_correlation_management'),
    path('wizard/correlations/apply/', wizard_views.wizard_manage_correlations, name='wizard_manage_correlations_apply'),
    path('wizard/select-analyses/', wizard_views.wizard_select_analyses, name='wizard_select_analyses'),
    path('wizard/run-analyses/', wizard_views.wizard_run_analyses, name='wizard_run_analyses'),
    
    # Classic dashboard (backup)
    path('classic/', views.dashboard, name='dashboard'),
    path('load/titanic/', views.load_titanic, name='load_titanic'),
    path('load/iris/', views.load_iris, name='load_iris'),
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('selection/', views.set_target_and_features, name='set_target_and_features'),
    path('sampling/', views.set_sampling, name='set_sampling'),
    path('types/', views.set_manual_types, name='set_manual_types'),
    path('reset/', views.reset_session, name='reset_session'),

    path('run/eda/descriptive/', views.run_descriptive, name='run_descriptive'),
    path('run/eda/correlation/', views.run_correlation, name='run_correlation'),
    path('run/eda/distribution/', views.run_distribution, name='run_distribution'),
    path('run/eda/outliers/', views.run_outliers, name='run_outliers'),
    path('run/eda/categorical/', views.run_categorical, name='run_categorical'),
    path('run/eda/stat_tests/', views.run_stat_tests, name='run_stat_tests'),

    path('run/ml/train/', views.run_ml_train, name='run_ml_train'),
    path('run/ml/cluster/', views.run_clustering, name='run_clustering'),

    path('run/text/', views.run_text, name='run_text'),
    path('run/time_series/', views.run_time_series, name='run_time_series'),

    path('predict/', views.predict, name='predict'),

    path('export/session/', views.export_session_json, name='export_session_json'),
    path('export/report.html', views.export_report_html, name='export_report_html'),
    path('export/report.pdf', views.export_report_pdf, name='export_report_pdf'),
    path('export/data/', views.export_data, name='export_data'),
    path('export/model/', views.export_model_bundle, name='export_model_bundle'),
    path('export/code/', views.export_python_code_last, name='export_python_code_last'),
    path('export/visuals.zip', views.export_visualizations_zip, name='export_visualizations_zip'),
    path('export/bundle.zip', views.export_bundle_zip, name='export_bundle_zip'),

    path('inline/visuals/generate/', views.generate_inline_visuals, name='generate_inline_visuals'),
    path('inline/visuals/<str:filename>', views.inline_visual, name='inline_visual'),
]
