"""
tracking.py -- optional MLflow experiment tracking.

    from tracking import log_run
    log_run("train-synthetic", params={...}, metrics={...}, artifacts=[path, ...])

A no-op (with one printed note) if mlflow is not installed OR if logging fails
for any reason, so training never depends on it. By default runs go to a local
SQLite file (project/mlflow.db; needs `pip install mlflow-skinny sqlalchemy
alembic`); set MLFLOW_TRACKING_URI to use a server instead. Browse with
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.
"""
import os

_NOTED = False


def log_run(run_name, params=None, metrics=None, artifacts=None, experiment="pdm"):
    global _NOTED
    try:
        import mlflow
        uri = os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///" + os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "mlflow.db"))
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({k: str(v) for k, v in (params or {}).items()})
            flat = {k: float(v) for k, v in (metrics or {}).items() if v is not None}
            if flat:
                mlflow.log_metrics(flat)
            for a in artifacts or []:
                if os.path.exists(a):
                    mlflow.log_artifact(a)
            return run.info.run_id
    except Exception as e:                       # never let tracking break training
        if not _NOTED:
            print(f"NOTE: MLflow tracking skipped ({type(e).__name__}: {str(e)[:120]})")
            _NOTED = True
        return None
