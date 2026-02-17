import os
import csv
from datetime import datetime

import pandas as pd
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model, scaler, and type encoder
model = pickle.load(open("models/fraud_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
type_encoder = pickle.load(open("models/type_encoder.pkl", "rb"))

LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "transactions_log.csv")
os.makedirs(LOG_DIR, exist_ok=True)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        account_id = request.form["account_id"]
        step_val = float(request.form["step"])
        type_encoded = int(request.form["type_encoded"])  # already encoded 0–4
        amount_val = float(request.form["amount"])
        oldbalanceOrg = float(request.form["oldbalanceOrg"])
        newbalanceOrig = float(request.form["newbalanceOrig"])
        oldbalanceDest = float(request.form["oldbalanceDest"])
        newbalanceDest = float(request.form["newbalanceDest"])

        # Build feature vector in the same order as in training
        feature_cols = [
            "step",
            "type_encoded",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        ]

        input_df = pd.DataFrame(
            [[step_val, type_encoded, amount_val,
              oldbalanceOrg, newbalanceOrig,
              oldbalanceDest, newbalanceDest]],
            columns=feature_cols,
        )

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1]
        proba_percent = round(proba * 100, 2)

        if prediction == 1:
            result = f"⚠️ Fraudulent Transaction (risk score: {proba_percent}%)"
        else:
            result = f"✅ Legitimate Transaction (fraud risk: {proba_percent}%)"

        # Log to CSV for account-level analysis
        log_row = [
            datetime.now().isoformat(timespec="seconds"),
            account_id,
            step_val,
            type_encoded,
            amount_val,
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,
            newbalanceDest,
            int(prediction),
            proba_percent,
        ]

        file_exists = os.path.isfile(LOG_PATH)
        with open(LOG_PATH, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "account_id",
                    "step",
                    "type_encoded",
                    "amount",
                    "oldbalanceOrg",
                    "newbalanceOrig",
                    "oldbalanceDest",
                    "newbalanceDest",
                    "is_fraud",
                    "fraud_score",
                ])
            writer.writerow(log_row)

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        print("Error during prediction:", e)
        return render_template("index.html", prediction_text="Error in input or prediction")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/account-summary", methods=["GET"])
def account_summary():
    account_id = request.args.get("account_id", default=None, type=str)

    if not os.path.isfile(LOG_PATH):
        return render_template(
            "account_summary.html",
            summary=None,
            transactions=None,
            not_first_time=bool(account_id),
        )

    df_logs = pd.read_csv(LOG_PATH)

    if not account_id:
        return render_template(
            "account_summary.html",
            summary=None,
            transactions=None,
            not_first_time=False,
        )

    df_acc = df_logs[df_logs["account_id"] == account_id]

    if df_acc.empty:
        return render_template(
            "account_summary.html",
            summary=None,
            transactions=None,
            not_first_time=True,
        )

    total_txn = int(df_acc.shape[0])
    fraud_txn = int(df_acc[df_acc["is_fraud"] == 1].shape[0])
    avg_fraud_score = round(df_acc["fraud_score"].mean(), 2)
    is_high_risk = fraud_txn >= 3 or avg_fraud_score > 60

    df_recent = df_acc.sort_values("timestamp", ascending=False).head(10)

    transactions = [
        {
            "timestamp": row["timestamp"],
            "time": row["step"],
            "amount": row["amount"],
            "is_fraud": int(row["is_fraud"]),
            "fraud_score": row["fraud_score"],
        }
        for _, row in df_recent.iterrows()
    ]

    summary = {
        "account_id": account_id,
        "total_txn": total_txn,
        "fraud_txn": fraud_txn,
        "avg_fraud_score": avg_fraud_score,
        "is_high_risk": is_high_risk,
    }

    return render_template(
        "account_summary.html",
        summary=summary,
        transactions=transactions,
        not_first_time=True,
    )


if __name__ == "__main__":
    app.run(debug=True)
