"""
Time Series Analysis module for DataAnalyzer 2.0
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import time
from statsmodels.tsa.stattools import adfuller, acf, pacf

def analyze_time_series(df: pd.DataFrame, date_column: str, value_column: str,
                       params: Dict = {}) -> Dict:
    """
    Analyse de séries temporelles basique
    
    Args:
        df: DataFrame
        date_column: Nom de la colonne date
        value_column: Nom de la colonne valeur
        params: Paramètres additionnels
        
    Returns:
        Dictionnaire des résultats
    """
    start_time = time.time()
    
    try:
        # Convertir en datetime / numérique
        df_ts = df[[date_column, value_column]].copy()
        df_ts[date_column] = pd.to_datetime(df_ts[date_column], errors='coerce')
        df_ts[value_column] = pd.to_numeric(df_ts[value_column], errors='coerce')
        df_ts = df_ts.dropna().sort_values(date_column).set_index(date_column)

        # Série
        ts = df_ts[value_column]
        if len(ts) < 10:
            return {
                'success': False,
                'error': 'Série temporelle insuffisante (>=10 observations recommandées).',
                'execution_time': time.time() - start_time
            }
        
        # Tendance (régression linéaire simple)
        x = np.arange(len(ts))
        y = ts.values
        coeffs = np.polyfit(x, y, 1)
        trend = 'Croissante' if coeffs[0] > 0 else 'Décroissante'
        
        # Stationnarité (ADF)
        adf_out = None
        try:
            adf_stat, adf_p, _, _, crit, _ = adfuller(ts.values, autolag='AIC')
            adf_out = {
                'adf_stat': float(adf_stat),
                'p_value': float(adf_p),
                'critical_values': {str(k): float(v) for k, v in crit.items()},
            }
        except Exception:
            adf_out = None

        # ACF / PACF
        lags = int(params.get('lags', 40))
        lags = max(10, min(lags, 200, len(ts) - 1))
        acf_vals = None
        pacf_vals = None
        try:
            acf_vals = [float(x) for x in acf(ts.values, nlags=lags, fft=True)]
            pacf_vals = [float(x) for x in pacf(ts.values, nlags=lags, method='ywm')]
        except Exception:
            acf_vals = None
            pacf_vals = None

        # Baseline moyenne mobile
        ma_window = int(params.get('ma_window', 7))
        ma_window = max(2, min(ma_window, 365, len(ts)))
        ma = None
        try:
            ma = ts.rolling(window=ma_window, min_periods=1).mean()
        except Exception:
            ma = None
        
        results = {
            'success': True,
            'results': {
                'n_observations': len(ts),
                'date_range': [str(ts.index.min()), str(ts.index.max())],
                'trend': trend,
                'mean': float(ts.mean()),
                'std': float(ts.std()),
                'min': float(ts.min()),
                'max': float(ts.max()),
                'adf': adf_out,
                'lags': lags,
                'acf': acf_vals,
                'pacf': pacf_vals,
                'ma_window': ma_window,
            },
            'visualizations': ['time_series_plot', 'acf_pacf'],
            'explanations': {
                'method': "Analyse de séries temporelles (tendance + stationnarité + autocorrélations)",
                'interpretation': """
                - Tendance: Direction générale de la série
                - ADF: test de stationnarité (p-value faible => stationnaire)
                - ACF/PACF: dépendances temporelles (lags)
                """,
                'warnings': [
                    'Adapter les lags et la fenêtre MA selon la fréquence de la série'
                ]
            },
            'python_code': f"""
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf

df = pd.read_csv('votre_fichier.csv')
df['{date_column}'] = pd.to_datetime(df['{date_column}'])
df = df.sort_values('{date_column}').set_index('{date_column}')
ts = df['{value_column}'].dropna()

x = np.arange(len(ts))
coeffs = np.polyfit(x, ts.values, 1)
trend = 'Croissante' if coeffs[0] > 0 else 'Décroissante'
print('Trend:', trend)
print('Mean:', ts.mean())

# ADF
adf_stat, p_value, _, _, crit, _ = adfuller(ts.values, autolag='AIC')
print('ADF p-value:', p_value)

# ACF/PACF
lags = {lags}
acf_vals = acf(ts.values, nlags=lags)
pacf_vals = pacf(ts.values, nlags=lags, method='ywm')
print('ACF[1:5]:', acf_vals[1:6])
""",
            'execution_time': time.time() - start_time
        }
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'execution_time': time.time() - start_time
        }
