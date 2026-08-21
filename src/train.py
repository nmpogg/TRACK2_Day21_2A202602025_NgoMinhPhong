import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
import inspect
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

EVAL_THRESHOLD = 0.70


def setup_mlflow():
    """
    Bonus 1: Ho tro tracking MLflow tu xa voi DagsHub neu duoc cau hinh.
    Neu co DAGSHUB_USERNAME va DAGSHUB_TOKEN thi tu dong ket noi DagsHub.
    """
    dagshub_user = os.environ.get("DAGSHUB_USERNAME")
    dagshub_token = os.environ.get("DAGSHUB_TOKEN")
    dagshub_repo = os.environ.get("DAGSHUB_REPO", "TRACK2_Day21_2A202602025_NgoMinhPhong")

    if dagshub_user and dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        tracking_uri = f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        print(f"[MLflow] Remote Tracking URI: {tracking_uri}")
    elif "MLFLOW_TRACKING_URI" in os.environ:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


def build_model(model_type: str, params: dict):
    """
    Bonus 2: Khoi tao mo hinh dua tren model_type va loc cac tham so phu hop.
    Ho tro: random_forest, gradient_boosting, logistic_regression.
    """
    model_type = model_type.lower()
    
    if model_type == "gradient_boosting":
        valid_args = inspect.signature(GradientBoostingClassifier.__init__).parameters
        filtered_params = {k: v for k, v in params.items() if k in valid_args}
        return GradientBoostingClassifier(random_state=42, **filtered_params)
    elif model_type == "logistic_regression":
        valid_args = inspect.signature(LogisticRegression.__init__).parameters
        filtered_params = {k: v for k, v in params.items() if k in valid_args}
        return LogisticRegression(random_state=42, max_iter=1000, **filtered_params)
    else:  # default: random_forest
        valid_args = inspect.signature(RandomForestClassifier.__init__).parameters
        filtered_params = {k: v for k, v in params.items() if k in valid_args}
        return RandomForestClassifier(random_state=42, **filtered_params)


def check_data_drift(y_train: pd.Series) -> dict:
    """
    Bonus 5: Kiem tra phan phoi nhan trong tap huan luyen.
    Neu bat ky lop nao chiem < 10% tong mau, in canh bao ro rang.
    """
    class_ratios = y_train.value_counts(normalize=True).to_dict()
    label_distribution = {}

    print("--- PHAN PHOI NHAN (CLASS DISTRIBUTION) ---")
    for cls, ratio in sorted(class_ratios.items()):
        label_distribution[f"class_{cls}_ratio"] = round(float(ratio), 4)
        print(f"Lop {cls}: {ratio:.2%} ({int(y_train.value_counts()[cls])} mau)")
        if ratio < 0.10:
            print(f"[CANH BAO DATA IMBALANCE] Lop {cls} chi chiem {ratio:.2%} (< 10%) tong mau huan luyen!")

    return label_distribution


def generate_report(y_eval, preds, acc: float, f1: float, model_type: str):
    """
    Bonus 3: Tao bao cao hieu suat chi tiet va ma tran nham lan (outputs/report.txt).
    """
    os.makedirs("outputs", exist_ok=True)
    report_text = classification_report(y_eval, preds, digits=4, zero_division=0)
    cm = confusion_matrix(y_eval, preds)

    with open("outputs/report.txt", "w", encoding="utf-8") as f:
        f.write("=====================================================\n")
        f.write("      BAO CAO DANH GIA HIEU SUAT MO HINH MLOPS       \n")
        f.write("=====================================================\n\n")
        f.write(f"Model Architecture : {model_type.upper()}\n")
        f.write(f"Accuracy           : {acc:.4f}\n")
        f.write(f"F1-Score (Weighted): {f1:.4f}\n\n")
        f.write("--- CLASSIFICATION REPORT (PRECISION / RECALL / F1) ---\n")
        f.write(report_text)
        f.write("\n\n--- CONFUSION MATRIX ---\n")
        f.write(str(cm))
        f.write("\n")
    print("Da luu bao cao hieu suat chi tiet vao outputs/report.txt")


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.
    """
    setup_mlflow()

    # Doc du lieu
    df_train = pd.read_csv(data_path)
    df_eval  = pd.read_csv(eval_path)

    # Tach dac trung va nhan
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval  = df_eval.drop(columns=["target"])
    y_eval  = df_eval["target"]

    # Bonus 5: Kiem tra data imbalance
    label_distribution = check_data_drift(y_train)

    # Bonus 2: Lay model_type va khoi tao mo hinh
    model_type = params.get("model_type", "random_forest")
    model = build_model(model_type, params)

    with mlflow.start_run():
        # Ghi nhan sieu tham so
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_type)

        # Huan luyen mo hinh
        model.fit(X_train, y_train)

        # Du doan va tinh metrics
        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")

        # Ghi metrics vao MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        for k, v in label_distribution.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(model, "model")

        print(f"[{model_type.upper()}] Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Bonus 3: Tao bao cao hieu suat chi tiet
        generate_report(y_eval, preds, acc, f1, model_type)

        # Luu outputs/metrics.json
        os.makedirs("outputs", exist_ok=True)
        metrics_data = {
            "accuracy": acc,
            "f1_score": f1,
            "model_type": model_type,
            "label_distribution": label_distribution,
        }
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=2)

        # Luu mo hinh ra models/model.pkl
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
