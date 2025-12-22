#!/usr/bin/env python3
"""
Test script for DataAnalyzer 2.0
Validates core functionality and scientific rules
"""
import sys
sys.path.insert(0, '.')

from modules.data_loader import load_data
from modules.data_profiler import profile_dataframe
from modules.ml_models import train_classification_model
from modules.text_analysis import analyze_text
from modules.time_series import analyze_time_series
from utils.validation import detect_problem_type, validate_target_not_in_features
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from sklearn.datasets import load_iris

def test_data_loading():
    """Test 1: Data loading"""
    print("\n" + "="*60)
    print("TEST 1: Data Loading")
    print("="*60)
    
    df, error = load_data('modules/data/Titanic-Dataset.csv')
    assert df is not None, f"Failed to load data: {error}"
    assert df.shape == (891, 12), f"Unexpected shape: {df.shape}"
    print(f"OK: Titanic dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df

def test_profiling(df):
    """Test 2: Data profiling"""
    print("\n" + "="*60)
    print("TEST 2: Data Profiling")
    print("="*60)
    
    profile = profile_dataframe(df)
    assert profile is not None, "Profiling failed"
    assert 'columns' in profile, "Missing columns info"
    assert 'quality_metrics' in profile, "Missing quality metrics"
    print(f"OK: Profile generated with {len(profile['columns'])} columns analyzed")
    return profile

def test_problem_type_detection(df):
    """Test 3: Problem type detection"""
    print("\n" + "="*60)
    print("TEST 3: Problem Type Detection")
    print("="*60)
    
    target = 'Survived'
    problem_type, description = detect_problem_type(df[target])
    assert problem_type == 'binary_classification', f"Wrong type: {problem_type}"
    print(f"OK: Target '{target}' correctly detected as: {description}")
    return target

def test_target_feature_separation(df, target):
    """Test 4: RÈGLE 1 - Target/Feature Separation"""
    print("\n" + "="*60)
    print("TEST 4: RÈGLE 1 - Target/Feature Separation")
    print("="*60)
    
    # Correct case: target NOT in features
    features_correct = ['Pclass', 'Sex', 'Age', 'Fare']
    is_valid, message = validate_target_not_in_features(features_correct, target)
    assert is_valid, "Valid features marked as invalid"
    print(f"OK: Validation passed: Features WITHOUT target")
    
    # Wrong case: target IN features
    features_wrong = ['Pclass', 'Sex', 'Survived', 'Fare']
    is_valid, message = validate_target_not_in_features(features_wrong, target)
    assert not is_valid, "Invalid features not detected"
    print(f"OK: Validation passed: Features WITH target correctly rejected")
    print(f"   Error message: {message}")
    
    return features_correct

def test_ml_training(df, target, features):
    """Test 5: ML Model Training"""
    print("\n" + "="*60)
    print("TEST 5: ML Model Training")
    print("="*60)
    
    # Correct training
    params = {
        'test_size': 0.2,
        'random_state': 42,
        'scale': True,
        'models': ['logistic']
    }
    
    results = train_classification_model(df, target, features, params)
    assert results['success'], f"Training failed: {results.get('error')}"
    print(f"OK: Model trained successfully")
    print(f"   Best model: {results['results']['best_model']}")
    print(f"   F1-Score: {results['results']['best_score']:.3f}")
    
    # Wrong training (target in features)
    features_wrong = features + [target]
    results_wrong = train_classification_model(df, target, features_wrong, params)
    assert not results_wrong['success'], "Training with target in features should fail"
    print(f"OK: Training with target in features correctly rejected")


def test_iris_classification():
    """Test 6: IRIS classification (sanity check)"""
    print("\n" + "="*60)
    print("TEST 6: IRIS Classification")
    print("="*60)

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    # sklearn fournit déjà une colonne cible 'target'
    target = 'target'
    features = [c for c in df.columns if c != target]

    params = {
        'test_size': 0.2,
        'random_state': 42,
        'scale': True,
        'models': ['logistic']
    }
    results = train_classification_model(df, target, features, params)
    assert results['success'], f"IRIS training failed: {results.get('error')}"
    print("OK: IRIS model trained")


def test_time_series_activation_and_run():
    """Test 7: date column detected + time series run"""
    print("\n" + "="*60)
    print("TEST 7: Time Series (date detection + run)")
    print("="*60)

    df = pd.DataFrame({
        'ds': pd.date_range('2024-01-01', periods=50, freq='D'),
        'y': [float(i) for i in range(50)],
    })
    prof = profile_dataframe(df)
    assert prof['columns']['ds']['type'] == 'date', f"Expected date type, got {prof['columns']['ds']['type']}"

    results = analyze_time_series(df, date_column='ds', value_column='y', params={})
    assert results.get('success'), f"Time series failed: {results.get('error')}"
    print("OK: Time series analysis executed")


def test_text_tfidf_activation_and_run():
    """Test 8: text column detected + TF-IDF run"""
    print("\n" + "="*60)
    print("TEST 8: Text (type detection + TF-IDF)")
    print("="*60)

    df = pd.DataFrame({
        'text': [
            'bonjour monde',
            'analyse de texte tfidf',
            'bonjour analyse',
            'modele tfidf simple'
        ]
    })
    prof = profile_dataframe(df)
    assert prof['columns']['text']['type'] in {'text', 'categorical'}, f"Unexpected text type: {prof['columns']['text']['type']}"

    results = analyze_text(df, text_column='text', params={'method': 'tfidf', 'top_k': 10})
    assert results.get('success'), f"TF-IDF analysis failed: {results.get('error')}"
    assert 'results' in results and 'tfidf_top_terms' in results['results'], "Missing tfidf_top_terms in results"
    print("OK: TF-IDF analysis executed")

def run_all_tests():
    """Run all validation tests"""
    print("\n" + "="*60)
    print("DATAANALYZER 2.0 - VALIDATION TESTS")
    print("="*60)
    
    try:
        df = test_data_loading()
        profile = test_profiling(df)
        target = test_problem_type_detection(df)
        features = test_target_feature_separation(df, target)
        test_ml_training(df, target, features)
        test_iris_classification()
        test_time_series_activation_and_run()
        test_text_tfidf_activation_and_run()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        print("Data loading: OK")
        print("Data profiling: OK")
        print("Problem type detection: OK")
        print("Target/Feature separation (RÈGLE 1): OK")
        print("ML training validation: OK")
        print("IRIS classification: OK")
        print("Time series (date detection + run): OK")
        print("Text TF-IDF: OK")
        print("\nDataAnalyzer 2.0 is ready to use!")
        return 0
        
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
