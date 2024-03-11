from pydantic_settings import BaseSettings

seed = 2024


class model_settings(BaseSettings):
    data_path: str = "data/HR-Employee-Attrition.csv"
    user: str = "mlflow"
    pw: str = "mlflow"
    db: str = "mlflow"
    host: str = "mlflow.aws-region.rds.amazonaws.com"
    target: str = "Attrition"
    connection_string: str = f"postgresql://{user}:{pw}@{host}:5432/{db}"
    artifact_uri: str = "s3://mlflow"
    selected_features: list = [
        "BusinessTravel",
        "DailyRate",
        "Department",
        "DistanceFromHome",
        "Education",
        "EducationField",
        "EnvironmentSatisfaction",
        "Gender",
        "JobInvolvement",
        "JobLevel",
        "JobRole",
        "JobSatisfaction",
        "MaritalStatus",
        "MonthlyIncome",
        "NumCompaniesWorked",
        "OverTime",
        "StockOptionLevel",
        "WorkLifeBalance",
    ]
    params: dict = {
        "boosting_type": "gbdt",
        "colsample_bytree": 0.8,
        "learning_rate": 0.1,
        "max_depth": -1,
        "metric": "binary_logloss",
        "n_jobs": -1,
        "num_leaves": 31,
        "objective": "binary",
        "random_state": seed,
        "subsample": 0.5,
        "verbosity": -1,
        "subsample_freq": 5,
    }
