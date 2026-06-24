# Databricks notebook source

# COMMAND ----------
import sys
sys.path.append('/Workspace/Users/pidkalwar.rahul@gmail.com/.bundle/databricks-churn-mlops/prod/files')

import mlflow
from datetime import datetime
from src.features import load_data, engineer_features, get_train_test_split
from src.train    import train_model, log_model_to_mlflow
from src.evaluate import evaluate_model
from src.promote  import promote_if_better
from src.drift    import check_drift

print(f"=== Pipeline Started: {datetime.now()} ===")

# COMMAND ----------
# Task 1 — Check drift first
drift = check_drift(spark, threshold=0.2)

if not drift:
    print("\n✅ No drift detected — skipping retraining")
    dbutils.notebook.exit("NO_DRIFT")

# COMMAND ----------
# Task 2 — Drift detected, retrain
print("\n🚨 Drift detected — starting retraining")

mlflow.set_experiment("/Users/pidkalwar.rahul@gmail.com/churn_prediction")

df                               = load_data(spark)
df                               = engineer_features(df)
X_train, X_test, y_train, y_test = get_train_test_split(df)

# COMMAND ----------
# Task 3 — Train + evaluate + promote
model   = train_model(X_train, y_train)
metrics = evaluate_model(model, X_test, y_test)

params  = {
    "n_estimators"     : 200,
    "max_depth"        : 10,
    "min_samples_split": 5,
    "trigger"          : "drift_detected"
}

run_name = f"drift_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
run_id   = log_model_to_mlflow(model, params, metrics, X_train, run_name)

# COMMAND ----------
promoted = promote_if_better(run_id, metrics["roc_auc"])
print(f"\nPromoted: {promoted}")
print(f"=== Pipeline Completed: {datetime.now()} ===")