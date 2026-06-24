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

print(f"=== Retraining Job Started: {datetime.now()} ===")

# COMMAND ----------
mlflow.set_experiment("/Users/pidkalwar.rahul@gmail.com/churn_prediction")

df                               = load_data(spark)
df                               = engineer_features(df)
X_train, X_test, y_train, y_test = get_train_test_split(df)

# COMMAND ----------
model   = train_model(X_train, y_train)
metrics = evaluate_model(model, X_test, y_test)

params  = {
    "n_estimators"     : 200,
    "max_depth"        : 10,
    "min_samples_split": 5,
    "trigger"          : "cicd"
}

run_name = f"cicd_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
run_id   = log_model_to_mlflow(model, params, metrics, X_train, run_name)

# COMMAND ----------
promoted = promote_if_better(run_id, metrics["roc_auc"])
print(f"Promoted: {promoted}")
print(f"=== Job Completed: {datetime.now()} ===")