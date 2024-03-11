import warnings

warnings.filterwarnings("ignore")
from datetime import datetime
import mlflow
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import shap
from contextlib import contextmanager
import tempfile
import pickle

seed = 2024


def read_data(path, col_list, target):
    data = pd.read_csv(path)
    data[target] = data[target].map({"Yes": 1, "No": 0})
    constant_cols = data.nunique()[data.nunique() == 1].keys().tolist()
    data.drop(constant_cols, axis=1, inplace=True)
    cols = [c for c in data.columns if c in col_list + [target]]

    return data[cols]


def replace_categories(df, var, target):
    ordered_labels = (
        df.groupby([var])[target].mean().to_frame().sort_values(target).index
    )
    ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}
    df[var] = df[var].map(ordinal_label)

    return ordinal_label


def preprocessing(data, target: str):
    cat_cols = [cat for cat in data.select_dtypes("O").columns.tolist()]

    df = data.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    label_list = []
    for var in cat_cols:
        lbl = replace_categories(df, var, target)
        label_list.append((var, lbl))

    return df, dict(label_list)


def data_split(data, target: str):
    train_cols = [c for c in data.columns if c not in [target]]

    x_train, x_test, y_train, y_test = train_test_split(
        data[train_cols],
        data[target],
        test_size=0.2,
        random_state=seed,
        stratify=data[target],
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train[train_cols], y_train, test_size=0.2, random_state=seed, stratify=y_train
    )

    return x_train, x_test, x_val, y_train, y_test, y_val, train_cols, target


def scorer(y_true, y_pred, is_return=False):
    if is_return:
        return [
            round(accuracy_score(y_true, np.round(y_pred)), 4),
            round(f1_score(y_true, np.round(y_pred)), 4),
            round(recall_score(y_true, np.round(y_pred)), 4),
            round(precision_score(y_true, np.round(y_pred)), 4),
            round(roc_auc_score(y_true, y_pred), 4),
        ]
    else:
        print("F1 (macro): {:.4f}".format(f1_score(y_true, np.round(y_pred)))),
        print("Accuracy: {:.4f}".format(accuracy_score(y_true, np.round(y_pred)))),
        print("Recall: {:.4f}".format(recall_score(y_true, np.round(y_pred))))
        print("Precision: {:.4f}".format(precision_score(y_true, np.round(y_pred))))
        print(classification_report(y_true, np.round(y_pred)))
        print(confusion_matrix(y_true, np.round(y_pred)))


def lgb_fit_mlflow(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train,
    y_val,
    y_test,
    params: dict,
    cols: list,
    cat_cols: list = None,
    n_tree: int = 1000,
    tracking_uri: str = None,
    artifact_uri: str = None,
):

    dt = datetime.today().strftime("%Y%m%d_%H%M")
    mlflow.lightgbm.autolog(
        log_input_examples=True,
        log_models=True,
        log_datasets=True,
        registered_model_name="hr_attrition_model",
        silent=False,
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(artifact_uri)

    exp_name = "hr_attrition"
    # https://github.com/mlflow/mlflow/issues/2464#issuecomment-879373585
    try:
        experiment = mlflow.get_experiment_by_name(exp_name)
        experiment_id = experiment.experiment_id
    except AttributeError:
        experiment_id = mlflow.create_experiment(
            exp_name, artifact_location=artifact_uri + f"/{exp_name}"
        )

    with mlflow.start_run(
        nested=False,
        experiment_id=mlflow.set_experiment(exp_name).experiment_id,
        run_name=f"train_{dt}",
        tags={"data": "hr-attrition", "model": "lightgbm"},
        description="HR Attrition Prediction Model",
    ):

        dtrain = lgb.Dataset(
            df_train[cols],
            y_train,
            categorical_feature=cat_cols,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            df_val[cols],
            y_val,
            categorical_feature=cat_cols,
            reference=dtrain,
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=n_tree,
            callbacks=[lgb.log_evaluation(period=10), lgb.early_stopping(25)],
            valid_sets=(dtrain, dval),
        )

        print("\n" * 2)

        # https://github.com/mlflow/mlflow/issues/3392#issuecomment-688573431
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_val)
        shap.summary_plot(shap_values, df_val, show=False)
        plt.savefig("summary_plot.png")
        mlflow.log_artifact("summary_plot.png")

        # first 200 samples
        plot = shap.force_plot(
            explainer.expected_value[1],
            shap_values[1][:200, :],
            df_val.iloc[:200, :],
            show=False,
        )
        # first sample
        fig = shap.force_plot(
            explainer.expected_value[1],
            shap_values[1][0, :],
            df_val.iloc[0, :],
            matplotlib=True,
            show=False,
        )

        log_shap_plot(plot, "force_plot.html")
        log_figure(fig, "force_plot.png")
        # log_pickle(explainer, "explainer.pkl")

        probs = model.predict(df_test[cols])
        scores = scorer(y_test, probs, True)
        mlflow.log_metric("test_accuracy", scores[0])
        mlflow.log_metric("test_f1", scores[1])
        mlflow.log_metric("test_recall", scores[2])
        mlflow.log_metric("test_precision", scores[3])
        mlflow.log_metric("test_roc", scores[4])
        tn, fp, fn, tp = confusion_matrix(y_test, np.round(probs)).ravel()
        mlflow.log_metric("test_true_positive", tp)
        mlflow.log_metric("test_true_negative", tn)
        mlflow.log_metric("test_false_positive", fp)
        mlflow.log_metric("test_false_negative", fn)
        print("\t" * 2, "Test Scores")
        scores = scorer(y_test, probs, False)
        cm = ConfusionMatrixDisplay.from_predictions(
            y_true=y_test, y_pred=np.round(probs), cmap="Blues"
        )
        cm.ax_.set_title("Test CM")
        mlflow.log_figure(cm.figure_, "test_confusion_matrix.png")
        print("-" * 50)
        print("\n")

        probs = model.predict(df_val[cols])
        scores = scorer(y_val, probs, True)
        mlflow.log_metric("val_accuracy", scores[0])
        mlflow.log_metric("val_f1", scores[1])
        mlflow.log_metric("val_recall", scores[2])
        mlflow.log_metric("val_precision", scores[3])
        mlflow.log_metric("val_roc", scores[4])
        tn, fp, fn, tp = confusion_matrix(y_val, np.round(probs)).ravel()
        mlflow.log_metric("val_true_positive", tp)
        mlflow.log_metric("val_true_negative", tn)
        mlflow.log_metric("val_false_positive", fp)
        mlflow.log_metric("val_false_negative", fn)

        print("\t" * 2, "Validation Scores")
        scores = scorer(y_val, probs, False)
        cm = ConfusionMatrixDisplay.from_predictions(
            y_true=y_val, y_pred=np.round(probs), cmap="Blues"
        )
        cm.ax_.set_title("Valid CM")
        mlflow.log_figure(cm.figure_, "valid_confusion_matrix.png")
        print("-" * 50)
        print("\n")
        mlflow.log_metric("feature_count", df_train.shape[1])
        mlflow.log_metric("row_count", df_train.shape[0])

        mlflow.end_run()

    return model


@contextmanager
def log_artifact_contextmanager(filename):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, filename)
        yield tmp_path
        mlflow.log_artifact(tmp_path)


def log_shap_plot(plot, filename):
    assert filename.endswith(".html")
    with log_artifact_contextmanager(filename) as tmp_path:
        shap.save_html(tmp_path, plot, full_html=True)


def log_figure(fig, filename, close=True):
    with log_artifact_contextmanager(filename) as tmp_path:
        fig.savefig(tmp_path)
        if close:
            plt.close(fig)


def log_pickle(obj, filename):
    with log_artifact_contextmanager(filename) as tmp_path:
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f)
