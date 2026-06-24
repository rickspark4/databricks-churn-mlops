import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(spark):
    df = spark.table("churn_raw").toPandas()
    print(f"Data loaded: {df.shape}")
    return df


def engineer_features(df):
    binary_cols = [
        'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in binary_cols:
        df[col] = (df[col] == 'Yes').astype(int)

    df['gender'] = (df['gender'] == 'Male').astype(int)
    df = pd.get_dummies(
        df, columns=['InternetService', 'Contract', 'PaymentMethod']
    )
    return df


def get_train_test_split(df):
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape} Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test