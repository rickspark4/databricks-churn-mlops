import mlflow
from mlflow.tracking import MlflowClient


def get_champion_auc():
    client = MlflowClient()
    champion = client.get_model_version_by_alias(
        name="churn_prediction_model",
        alias="champion"
    )
    run = client.get_run(champion.run_id)
    auc = run.data.metrics.get("roc_auc", 0)
    print(f"Current champion AUC: {auc}")
    return auc, champion.version


def promote_if_better(new_run_id, new_auc):
    client = MlflowClient()
    current_auc, current_version = get_champion_auc()

    if new_auc > current_auc:
        new_version = mlflow.register_model(
            model_uri=f"runs:/{new_run_id}/random_forest_model",
            name="churn_prediction_model"
        )
        client.set_registered_model_alias(
            name="churn_prediction_model",
            alias="champion",
            version=new_version.version
        )
        client.set_registered_model_alias(
            name="churn_prediction_model",
            alias="challenger",
            version=current_version
        )
        print(f"✅ Promoted version {new_version.version} as new champion")
        return True
    else:
        print(f"⚠️  New model AUC {new_auc} not better than champion {current_auc}")
        return False