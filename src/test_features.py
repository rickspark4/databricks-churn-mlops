import pandas as pd
import sys
sys.path.append('./src')
from features import engineer_features


def test_engineer_features_binary_encoding():
    # Create dummy dataframe
    df = pd.DataFrame({
        'Partner'        : ['Yes', 'No'],
        'Dependents'     : ['No', 'Yes'],
        'PhoneService'   : ['Yes', 'Yes'],
        'PaperlessBilling': ['No', 'Yes'],
        'MultipleLines'  : ['Yes', 'No'],
        'OnlineSecurity' : ['No', 'Yes'],
        'OnlineBackup'   : ['Yes', 'No'],
        'DeviceProtection': ['No', 'Yes'],
        'TechSupport'    : ['Yes', 'No'],
        'StreamingTV'    : ['No', 'Yes'],
        'StreamingMovies': ['Yes', 'No'],
        'gender'         : ['Male', 'Female'],
        'InternetService': ['DSL', 'Fiber optic'],
        'Contract'       : ['Month-to-month', 'One year'],
        'PaymentMethod'  : ['Electronic check', 'Mailed check'],
        'Churn'          : [1, 0]
    })

    result = engineer_features(df)

    # Binary columns should be 0 or 1
    assert result['Partner'].isin([0, 1]).all()
    assert result['gender'].isin([0, 1]).all()
    print("✅ All tests passed")


if __name__ == "__main__":
    test_engineer_features_binary_encoding()