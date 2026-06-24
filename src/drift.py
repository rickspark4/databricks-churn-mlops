import numpy as np
from mlflow.tracking import MlflowClient


def calculate_psi(expected, actual, bins=10):
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1
    )
    expected_counts = np.histogram(expected, breakpoints)[0]
    actual_counts   = np.histogram(actual,   breakpoints)[0]

    expected_pct = np.where(expected_counts == 0, 0.0001, expected_counts / len(expected))
    actual_pct   = np.where(actual_counts   == 0, 0.0001, actual_counts   / len(actual))

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(psi, 4)


def check_drift(spark, threshold=0.2):
    """
    Loads reference data and current data from Delta tables.
    Returns True if drift detected, False if stable.
    """
    # Load reference data — original training data
    reference = spark.table("churn_raw").toPandas()

    # Load current inference log — recent predictions
    current = spark.table("churn_inference_log").toPandas()

    # Features to monitor
    features_to_monitor = ["tenure", "MonthlyCharges", "TotalCharges"]

    print("=== Drift Detection Report ===")
    print(f"{'Feature':<25} {'PSI':>8}  {'Status'}")
    print("-" * 50)

    drift_detected = False

    for feature in features_to_monitor:
        # Skip if feature not in current data
        if feature not in current.columns:
            continue

        psi = calculate_psi(
            reference[feature],
            current[feature]
        )

        if psi < 0.1:
            status = "✅ Stable"
        elif psi < threshold:
            status = "⚠️  Monitor"
        else:
            status = "🚨 Drift!"
            drift_detected = True

        print(f"{feature:<25} {psi:>8}  {status}")

    print("-" * 50)
    if drift_detected:
        print("Result: DRIFT DETECTED — retraining required")
    else:
        print("Result: NO DRIFT — retraining not needed")

    return drift_detected