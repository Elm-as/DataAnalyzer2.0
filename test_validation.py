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
from utils.validation import detect_problem_type, validate_target_not_in_features
import warnings
warnings.filterwarnings('ignore')

def test_data_loading():
    """Test 1: Data loading"""
    print("\n" + "="*60)
    print("TEST 1: Data Loading")
    print("="*60)
    
    df, error = load_data('data/Titanic-Dataset.csv')
    assert df is not None, f"Failed to load data: {error}"
    assert df.shape == (891, 12), f"Unexpected shape: {df.shape}"
    print(f"✅ Titanic dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
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
    print(f"✅ Profile generated with {len(profile['columns'])} columns analyzed")
    return profile

def test_problem_type_detection(df):
    """Test 3: Problem type detection"""
    print("\n" + "="*60)
    print("TEST 3: Problem Type Detection")
    print("="*60)
    
    target = 'Survived'
    problem_type, description = detect_problem_type(df[target])
    assert problem_type == 'binary_classification', f"Wrong type: {problem_type}"
    print(f"✅ Target '{target}' correctly detected as: {description}")
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
    print(f"✅ Validation passed: Features WITHOUT target")
    
    # Wrong case: target IN features
    features_wrong = ['Pclass', 'Sex', 'Survived', 'Fare']
    is_valid, message = validate_target_not_in_features(features_wrong, target)
    assert not is_valid, "Invalid features not detected"
    print(f"✅ Validation passed: Features WITH target correctly rejected")
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
    print(f"✅ Model trained successfully")
    print(f"   Best model: {results['results']['best_model']}")
    print(f"   F1-Score: {results['results']['best_score']:.3f}")
    
    # Wrong training (target in features)
    features_wrong = features + [target]
    results_wrong = train_classification_model(df, target, features_wrong, params)
    assert not results_wrong['success'], "Training with target in features should fail"
    print(f"✅ Training with target in features correctly rejected")

def run_all_tests():
    """Run all validation tests"""
    print("\n" + "="*60)
    print("🧪 DATAANALYZER 2.0 - VALIDATION TESTS")
    print("="*60)
    
    try:
        df = test_data_loading()
        profile = test_profiling(df)
        target = test_problem_type_detection(df)
        features = test_target_feature_separation(df, target)
        test_ml_training(df, target, features)
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("✅ Data loading: OK")
        print("✅ Data profiling: OK")
        print("✅ Problem type detection: OK")
        print("✅ Target/Feature separation (RÈGLE 1): OK")
        print("✅ ML training validation: OK")
        print("\nDataAnalyzer 2.0 is ready to use! 🚀")
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
