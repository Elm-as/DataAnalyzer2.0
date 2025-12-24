from django import forms


class UploadDatasetForm(forms.Form):
    file = forms.FileField(required=True, widget=forms.ClearableFileInput(attrs={'class': 'form-control'}))
    separator = forms.ChoiceField(
        choices=[(',', 'Virgule (,)'), (';', 'Point-virgule (;)')],
        required=True,
        widget=forms.Select(attrs={'class': 'form-select'}),
    )


class TargetSelectionForm(forms.Form):
    target = forms.ChoiceField(required=True, widget=forms.Select(attrs={'class': 'form-select'}))

    def __init__(self, *args, **kwargs):
        columns = kwargs.pop('columns', [])
        super().__init__(*args, **kwargs)
        self.fields['target'].choices = [(c, c) for c in columns]


class FeatureSelectionForm(forms.Form):
    features = forms.MultipleChoiceField(
        required=False,
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input feature-checkbox'}),
        help_text="Décochez pour désactiver des variables.",
    )

    def __init__(self, *args, **kwargs):
        columns = kwargs.pop('columns', [])
        target = kwargs.pop('target', None)
        super().__init__(*args, **kwargs)
        cols = [c for c in columns if c and c != target]
        self.fields['features'].choices = [(c, c) for c in cols]


class SamplingForm(forms.Form):
    enabled = forms.BooleanField(required=False)
    method = forms.ChoiceField(
        required=False,
        choices=[('random', 'Aléatoire'), ('stratified', 'Stratifié (classification)')],
    )
    n_rows = forms.IntegerField(required=False, min_value=100, max_value=1_000_000)
    random_state = forms.IntegerField(required=False, min_value=0, max_value=1_000_000)


class CorrelationParamsForm(forms.Form):
    method = forms.ChoiceField(
        choices=[('pearson', 'Pearson'), ('spearman', 'Spearman')],
        required=True,
        widget=forms.Select(attrs={'class': 'form-select'}),
    )
    threshold = forms.FloatField(required=True, min_value=0.0, max_value=1.0, initial=0.0, widget=forms.NumberInput(attrs={'class': 'form-control'}))


class OutliersParamsForm(forms.Form):
    iqr_multiplier = forms.FloatField(required=True, min_value=0.5, max_value=10.0, initial=1.5, widget=forms.NumberInput(attrs={'class': 'form-control'}))


class DistributionParamsForm(forms.Form):
    bins = forms.IntegerField(required=True, min_value=5, max_value=200, initial=30, widget=forms.NumberInput(attrs={'class': 'form-control'}))


class MLParamsForm(forms.Form):
    train_size = forms.IntegerField(required=True, min_value=60, max_value=90, initial=80, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    random_state = forms.IntegerField(required=True, min_value=0, max_value=1_000_000, initial=42, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    scale = forms.BooleanField(required=False, initial=True)

    models = forms.MultipleChoiceField(
        required=False,
        widget=forms.SelectMultiple(attrs={'size': '6', 'class': 'form-select'}),
        choices=[
            ('logistic', 'Logistic Regression'),
            ('random_forest', 'Random Forest'),
            ('xgboost', 'XGBoost'),
            ('lightgbm', 'LightGBM'),
            ('linear', 'Linear Regression'),
            ('ridge', 'Ridge'),
            ('lasso', 'Lasso'),
        ],
    )


class TextAnalysisForm(forms.Form):
    text_column = forms.ChoiceField(required=True)
    method = forms.ChoiceField(required=True, choices=[('basic', 'Basique'), ('tfidf', 'TF-IDF')])
    top_k = forms.IntegerField(required=True, min_value=5, max_value=100, initial=20)
    ngram_max = forms.IntegerField(required=False, min_value=1, max_value=3, initial=1)
    max_features = forms.IntegerField(required=False, min_value=200, max_value=50000, initial=2000)
    stop_words = forms.ChoiceField(required=False, choices=[('', 'Aucun'), ('english', 'English')])
    similarity_top_n = forms.IntegerField(required=False, min_value=1, max_value=50, initial=10)

    def __init__(self, *args, **kwargs):
        columns = kwargs.pop('columns', [])
        super().__init__(*args, **kwargs)
        self.fields['text_column'].choices = [(c, c) for c in columns]


class TimeSeriesForm(forms.Form):
    date_column = forms.ChoiceField(required=True)
    value_column = forms.ChoiceField(required=True)

    def __init__(self, *args, **kwargs):
        date_columns = kwargs.pop('date_columns', [])
        value_columns = kwargs.pop('value_columns', [])
        super().__init__(*args, **kwargs)
        self.fields['date_column'].choices = [(c, c) for c in date_columns]
        self.fields['value_column'].choices = [(c, c) for c in value_columns]
