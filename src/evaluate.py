from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "roc_auc"  : round(roc_auc_score(y_test, y_prob), 4),
        "f1_score" : round(f1_score(y_test, y_pred), 4)
    }

    print("=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"{k:12}: {v}")

    return metrics


def calculate_psi(expected, actual, bins=10):
    import numpy as np
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