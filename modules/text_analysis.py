"""
Text Analysis module for DataAnalyzer 2.0
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import time
from collections import Counter

def analyze_text(df: pd.DataFrame, text_column: str, params: Dict = {}) -> Dict:
    """
    Analyse basique de texte
    
    Args:
        df: DataFrame
        text_column: Nom de la colonne texte
        params: Paramètres additionnels
        
    Returns:
        Dictionnaire des résultats
    """
    start_time = time.time()
    
    try:
        texts = df[text_column].dropna().astype(str)
        
        if len(texts) == 0:
            return {
                'success': False,
                'error': "Aucun texte trouvé",
                'execution_time': time.time() - start_time
            }
        
        # Statistiques basiques
        lengths = texts.str.len()
        word_counts = texts.str.split().str.len()
        
        # Mots les plus fréquents (simple tokenization)
        all_words = ' '.join(texts).lower().split()
        word_freq = Counter(all_words).most_common(20)
        
        results = {
            'success': True,
            'results': {
                'n_texts': len(texts),
                'avg_length': float(lengths.mean()),
                'avg_word_count': float(word_counts.mean()),
                'min_length': int(lengths.min()),
                'max_length': int(lengths.max()),
                'top_words': word_freq,
                'vocabulary_size': len(set(all_words))
            },
            'visualizations': ['word_cloud'],
            'explanations': {
                'method': "Analyse de texte basique (tokenization simple)",
                'interpretation': """
                - Longueur moyenne et nombre de mots
                - Mots les plus fréquents
                - Pour analyse avancée: TF-IDF, embeddings
                """,
                'warnings': [
                    'Analyse simple sans preprocessing avancé'
                ]
            },
            'execution_time': time.time() - start_time
        }
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'execution_time': time.time() - start_time
        }
