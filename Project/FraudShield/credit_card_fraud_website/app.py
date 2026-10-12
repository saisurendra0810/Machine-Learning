
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split

app = Flask(__name__)
DATA_PATH = Path(__file__).parent / "credit_card_fraud_2026.csv"

TARGET = "is_fraud"
ID_COL = "transaction_id"

FEATURES = [
    "amount_usd", "merchant_category", "card_type", "auth_method",
    "channel", "device_type", "is_foreign_transaction",
    "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd",
    "is_new_merchant", "used_vpn", "ip_country_mismatch",
    "billing_shipping_mismatch", "cvv_retry_count", "velocity_score",
    "time_of_day_hour", "day_of_week", "is_ai_generated_scam_attempt",
    "merchant_risk_score", "prior_disputes"
]

CATEGORICAL = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
NUMERICAL = [c for c in FEATURES if c not in CATEGORICAL]

df = pd.read_csv(DATA_PATH)
X = df[FEATURES].copy()
y = df[TARGET].astype(int)

# Convert boolean columns to integers.
for col in FEATURES:
    if X[col].dtype == bool:
        X[col] = X[col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), NUMERICAL),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), CATEGORICAL)
])

logistic_model = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
])
svm_model = Pipeline([
    ("prep", preprocessor),
    # probability=False avoids the expensive/deprecated SVC probability
    # calibration step. We convert the decision function to a stable
    # probability-like score below for ROC-AUC, Log Loss, MAE, MSE, RMSE, R².
    ("model", LinearSVC(class_weight="balanced", random_state=42, max_iter=5000))
])

logistic_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Isolation Forest works on the numeric representation only.
iso_preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), NUMERICAL),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), CATEGORICAL)
])
X_train_iso = iso_preprocessor.fit_transform(X_train)
iso_model = IsolationForest(
    n_estimators=250, contamination="auto", random_state=42, n_jobs=-1
)
iso_model.fit(X_train_iso)

def calculate_adjusted_r2(y_true, y_prob, n_predictors):
    """Adjusted R-squared calculated from probability estimates.

    n_predictors is the number of original model input features.  This is
    reported as an additional regression-style diagnostic for the binary
    target and probability output.
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_prob)
    denominator = n - n_predictors - 1
    if denominator <= 0:
        return float("nan")
    return 1 - ((1 - r2) * (n - 1) / denominator)


def regression_metrics(y_true, y_prob):
    """Regression-style metrics treating fraud probabilities as predictions."""
    mae = mean_absolute_error(y_true, y_prob)
    mse = mean_squared_error(y_true, y_prob)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_prob)
    adjusted_r2 = calculate_adjusted_r2(y_true, y_prob, len(FEATURES))
    return {
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "adjusted_r2": round(float(adjusted_r2), 4) if np.isfinite(adjusted_r2) else None,
    }


def classification_metrics(y_true, pred, prob):
    """Classification metrics for fraud labels and predicted probabilities."""
    return {
        "accuracy": round(accuracy_score(y_true, pred) * 100, 2),
        "precision": round(precision_score(y_true, pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_true, pred, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_true, pred, zero_division=0) * 100, 2),
        "roc_auc": round(roc_auc_score(y_true, prob) * 100, 2),
        "log_loss": round(float(log_loss(y_true, prob, labels=[0, 1])), 4),
        **regression_metrics(y_true, prob),
    }


def svm_probability(model, data):
    """Convert SVM decision scores to stable probability-like values.

    The SVM uses LinearSVC, which provides a decision score rather than
    predict_proba. The logistic transform gives a stable probability-like
    score suitable for the requested probability-based evaluation metrics.
    """
    decision = np.asarray(model.decision_function(data), dtype=float)
    decision = np.clip(decision, -50.0, 50.0)
    prob = 1.0 / (1.0 + np.exp(-decision))
    return np.clip(prob, 1e-7, 1.0 - 1e-7)


def metrics_for(model, is_svm=False):
    pred = model.predict(X_test)
    if is_svm:
        prob = svm_probability(model, X_test)
    else:
        prob = model.predict_proba(X_test)[:, 1]
    return classification_metrics(y_test, pred, prob)


def isolation_metrics():
    test_matrix = iso_preprocessor.transform(X_test)
    pred = iso_model.predict(test_matrix)
    fraud_pred = (pred == -1).astype(int)

    # Convert anomaly scores to a stable 0..1 probability-like value so
    # log-loss and regression-style metrics can be calculated.
    score = -iso_model.decision_function(test_matrix)
    score_min, score_max = float(score.min()), float(score.max())
    if score_max > score_min:
        anomaly_prob = (score - score_min) / (score_max - score_min)
    else:
        anomaly_prob = np.full_like(score, 0.5, dtype=float)
    anomaly_prob = np.clip(anomaly_prob, 1e-7, 1 - 1e-7)

    return classification_metrics(y_test, fraud_pred, anomaly_prob)


MODEL_METRICS = {
    "Logistic Regression": metrics_for(logistic_model),
    "SVM": metrics_for(svm_model, is_svm=True),
    "Isolation Forest": isolation_metrics(),
}

def clean_input(payload):
    row = {}
    for col in FEATURES:
        value = payload.get(col)
        if col in CATEGORICAL:
            row[col] = value if value not in (None, "") else np.nan
        else:
            try:
                row[col] = float(value) if value not in (None, "") else np.nan
            except (ValueError, TypeError):
                row[col] = np.nan
    out = pd.DataFrame([row])
    for col in FEATURES:
        if col in out and out[col].dtype == bool:
            out[col] = out[col].astype(int)
    return out

@app.route("/")
def index():
    fraud_count = int(df[TARGET].sum())
    legit_count = int(len(df) - fraud_count)
    return render_template(
        "index.html",
        total=len(df),
        fraud=fraud_count,
        legit=legit_count,
        fraud_rate=round(fraud_count / len(df) * 100, 2),
        metrics=MODEL_METRICS,
        categories={c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAL},
    )

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    row = clean_input(payload)

    lr_prob = float(logistic_model.predict_proba(row)[0, 1])
    svm_prob = float(svm_probability(svm_model, row)[0])

    iso_matrix = iso_preprocessor.transform(row)
    iso_pred = int(iso_model.predict(iso_matrix)[0])  # -1 anomaly, +1 normal
    anomaly_score = float(-iso_model.decision_function(iso_matrix)[0])
    iso_signal = 1.0 if iso_pred == -1 else 0.0

    # Balanced ensemble: supervised models + anomaly signal.
    ensemble_prob = 0.45 * lr_prob + 0.45 * svm_prob + 0.10 * iso_signal
    fraud = ensemble_prob >= 0.50

    risk = "High" if ensemble_prob >= 0.70 else ("Medium" if ensemble_prob >= 0.40 else "Low")

    return jsonify({
        "prediction": "Fraudulent" if fraud else "Legitimate",
        "risk": risk,
        "probability": round(ensemble_prob * 100, 2),
        "logistic_probability": round(lr_prob * 100, 2),
        "svm_probability": round(svm_prob * 100, 2),
        "isolation_anomaly": bool(iso_pred == -1),
        "anomaly_score": round(anomaly_score, 4),
    })

@app.route("/api/summary")
def summary():
    return jsonify({
        "total": int(len(df)),
        "fraud": int(df[TARGET].sum()),
        "legitimate": int((df[TARGET] == 0).sum()),
        "fraud_rate": round(float(df[TARGET].mean() * 100), 2),
        "metrics": MODEL_METRICS
    })

if __name__ == "__main__":
    app.run(debug=True)
