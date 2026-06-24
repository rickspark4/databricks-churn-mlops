import pandas as pd
import numpy as np
import sys
sys.path.append('./src')

from features import engineer_features, get_train_test_split
from drift import calculate_psi


def create_dummy_df():
    return pd.DataFrame({
        'Partner'         : ['Yes', 'No'],
        'Dependents'      : ['No', 'Yes'],
        'PhoneService'    : ['Yes', 'Yes'],
        'PaperlessBilling': ['No', 'Yes'],
        'MultipleLines'   : ['Yes', 'No'],
        'OnlineSecurity'  : ['No', 'Yes'],
        'OnlineBackup'    : ['Yes', 'No'],
        'DeviceProtection': ['No', 'Yes'],
        'TechSupport'     : ['Yes', 'No'],
        'StreamingTV'     : ['No', 'Yes'],
        'StreamingMovies' : ['Yes', 'No'],
        'gender'          : ['Male', 'Female'],
        'InternetService' : ['DSL', 'Fiber optic'],
        'Contract'        : ['Month-to-month', 'One year'],
        'PaymentMethod'   : ['Electronic check', 'Mailed check'],
        'Churn'           : [1, 0],
        'tenure'          : [12, 24],
        'MonthlyCharges'  : [50.0, 75.0],
        'TotalCharges'    : [600.0, 1800.0],
        'SeniorCitizen'   : [0, 1]
    })


def test_binary_encoding():
    df = create_dummy_df()
    result = engineer_features(df)
    assert result['Partner'].isin([0, 1]).all(), "Partner not binary encoded"
    assert result['gender'].isin([0, 1]).all(), "Gender not binary encoded"
    print("✅ test_binary_encoding passed")


def test_no_nulls_after_encoding():
    df = create_dummy_df()
    result = engineer_features(df)
    assert result.isnull().sum().sum() == 0, "Nulls found after encoding"
    print("✅ test_no_nulls_after_encoding passed")


def test_churn_column_preserved():
    df = create_dummy_df()
    result = engineer_features(df)
    assert 'Churn' in result.columns, "Churn column missing"
    print("✅ test_churn_column_preserved passed")


def test_train_test_split_sizes():
    df = create_dummy_df()
    df = engineer_features(df)
    # need at least 10 rows for stratified split
    df = pd.concat([df] * 10, ignore_index=True)
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    assert len(X_train) > len(X_test), "Train set should be larger than test set"
    assert len(X_train) + len(X_test) == len(df), "Split sizes don't add up"
    print("✅ test_train_test_split_sizes passed")


def test_psi_stable_data():
    # Same data should give PSI near 0
    data = np.random.normal(50, 10, 1000)
    psi = calculate_psi(data, data)
    assert psi < 0.1, f"PSI should be near 0 for identical data, got {psi}"
    print("✅ test_psi_stable_data passed")


def test_psi_drifted_data():
    # Very different data should give high PSI
    reference = np.random.normal(50, 5, 1000)
    drifted   = np.random.normal(200, 5, 1000)
    psi = calculate_psi(reference, drifted)
    assert psi > 0.2, f"PSI should be high for drifted data, got {psi}"
    print("✅ test_psi_drifted_data passed")


if __name__ == "__main__":
    test_binary_encoding()
    test_no_nulls_after_encoding()
    test_churn_column_preserved()
    test_train_test_split_sizes()
    test_psi_stable_data()
    test_psi_drifted_data()
    print("\n✅ All tests passed")