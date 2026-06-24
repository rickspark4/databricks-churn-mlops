import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Model trained successfully")
    return model


def log_model_to_mlflow(model, params, metrics, X_train, run_name):
    with mlflow.start_run(run_name=run_name) as run:

        # Log params
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log model
        # mlflow.sklearn.log_model(
        #     model,
        #     name="random_forest_model",
        #     input_example=X_train.iloc[:5]
        # )

        mlflow.sklearn.log_model(model,
        artifact_path="random_forest_model",
        input_example=X_train.iloc[:5])

        run_id = run.info.run_id
        print(f"Run logged: {run_id}")
        return run_id