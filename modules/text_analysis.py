"""
Text Analysis module for DataAnalyzer 2.0
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import time
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

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
        
        method = str(params.get('method', 'basic')).lower().strip()
        top_k = int(params.get('top_k', 20))
        ngram_max = int(params.get('ngram_max', 1))
        ngram_max = max(1, min(3, ngram_max))
        max_features = int(params.get('max_features', max(2000, top_k * 50)))
        max_features = max(200, min(50_000, max_features))
        stop_words = params.get('stop_words', None)
        if stop_words in ('none', '', False):
            stop_words = None

        compute_similarity = bool(params.get('compute_similarity', method == 'tfidf'))
        similarity_top_n = int(params.get('similarity_top_n', 10))
        similarity_top_n = max(1, min(50, similarity_top_n))
        similarity_max_docs = int(params.get('similarity_max_docs', 400))
        similarity_max_docs = max(50, min(1000, similarity_max_docs))

        # Statistiques basiques
        lengths = texts.str.len()
        word_counts = texts.str.split().str.len()
        
        # Mots les plus fréquents (simple tokenization)
        all_words = ' '.join(texts).lower().split()
        word_freq = Counter(all_words).most_common(top_k)

        tfidf_top_terms = []
        similarity_top_pairs = []
        tfidf_vocab_size = None
        if method == 'tfidf':
            vectorizer = TfidfVectorizer(
                lowercase=True,
                max_features=max_features,
                stop_words=stop_words,
                ngram_range=(1, ngram_max),
            )
            X = vectorizer.fit_transform(texts.tolist())
            tfidf_vocab_size = int(len(vectorizer.get_feature_names_out()))
            # Moyenne TF-IDF par terme
            mean_scores = np.asarray(X.mean(axis=0)).ravel()
            terms = np.array(vectorizer.get_feature_names_out())
            top_idx = np.argsort(mean_scores)[::-1][:top_k]
            tfidf_top_terms = [(str(terms[i]), float(mean_scores[i])) for i in top_idx]

            # Similarité cosine (top paires) sur un échantillon limité
            if compute_similarity:
                try:
                    # On limite le nombre de docs pour éviter O(n^2) trop lourd
                    if len(texts) > similarity_max_docs:
                        sample = texts.sample(n=similarity_max_docs, random_state=42)
                    else:
                        sample = texts
                    Xs = vectorizer.transform(sample.tolist())
                    # Similarité = produit scalaire car TF-IDF est déjà normalisé (par défaut)
                    sims = linear_kernel(Xs, Xs)
                    # Extraire top paires (triangle supérieur)
                    n = sims.shape[0]
                    candidates = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            candidates.append((float(sims[i, j]), i, j))
                    candidates.sort(reverse=True, key=lambda t: t[0])
                    top = candidates[:similarity_top_n]
                    idx_list = sample.index.tolist()
                    similarity_top_pairs = [
                        {
                            'score': float(s),
                            'row_i': int(idx_list[i]) if str(idx_list[i]).isdigit() else str(idx_list[i]),
                            'row_j': int(idx_list[j]) if str(idx_list[j]).isdigit() else str(idx_list[j]),
                        }
                        for s, i, j in top
                    ]
                except Exception:
                    similarity_top_pairs = []
        
        results = {
            'success': True,
            'results': {
                'n_texts': len(texts),
                'avg_length': float(lengths.mean()),
                'avg_word_count': float(word_counts.mean()),
                'min_length': int(lengths.min()),
                'max_length': int(lengths.max()),
                'top_words': word_freq,
                'vocabulary_size': len(set(all_words)),
                'method': method,
                'tfidf_top_terms': tfidf_top_terms,
                'tfidf_vocab_size': tfidf_vocab_size,
                'ngram_max': ngram_max,
                'max_features': max_features,
                'stop_words': stop_words,
                'similarity_top_pairs': similarity_top_pairs,
            },
            'visualizations': ['word_cloud'],
            'explanations': {
                'method': "Analyse de texte" if method == 'tfidf' else "Analyse de texte basique (tokenization simple)",
                'interpretation': """
                - Longueur moyenne et nombre de mots
                - Mots les plus fréquents
                - TF-IDF: importance des termes pondérée par leur rareté
                """,
                'warnings': [
                    'La tokenisation est volontairement simple'
                ]
            },
            'python_code': f"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_csv('votre_fichier.csv')
texts = df['{text_column}'].dropna().astype(str)

method = '{method}'
top_k = {top_k}

ngram_max = {ngram_max}
max_features = {max_features}
stop_words = {repr(stop_words)}

if method == 'tfidf':
    vectorizer = TfidfVectorizer(lowercase=True, max_features=max_features, ngram_range=(1, ngram_max), stop_words=stop_words)
    X = vectorizer.fit_transform(texts.tolist())
    mean_scores = X.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    top_idx = mean_scores.argsort()[::-1][:top_k]
    tfidf_top_terms = [(terms[i], float(mean_scores[i])) for i in top_idx]
    print(tfidf_top_terms)

    # Similarité (top paires)
    sims = linear_kernel(X, X)
    # ... (extraire le triangle supérieur selon vos besoins)
else:
    all_words = ' '.join(texts).lower().split()
    from collections import Counter
    print(Counter(all_words).most_common(top_k))
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
