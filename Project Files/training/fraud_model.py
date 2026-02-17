import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Make sure folders exist
os.makedirs("../models", exist_ok=True)
os.makedirs("../static", exist_ok=True)

# 1. Load ONLY first 100,000 rows of the online payments dataset (faster)
# File must be: project_root/data/payments.csv  [web:56]
df_main = pd.read_csv("../data/payments.csv").head(100000)

# Optional: append new data later for adaptive retraining
new_data_path = "../data/new_transactions.csv"
if os.path.isfile(new_data_path):
    df_new = pd.read_csv(new_data_path)
    df_main = pd.concat([df_main, df_new], ignore_index=True)

df = df_main.copy()

# 2. Encode 'type' (string -> numeric)  [web:56][web:62]
type_encoder = LabelEncoder()
df["type_encoded"] = type_encoder.fit_transform(df["type"])

# 3. Select features and target (these columns must exist in payments.csv)
feature_cols = [
    "step",
    "type_encoded",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
X = df[feature_cols]
y = df["isFraud"]

# 4. Handle class imbalance using SMOTE  [web:56][web:59]
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# 6. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Small Random Forest model (fast)  [web:54][web:59]
rf = RandomForestClassifier(
    n_estimators=40,         # small number of trees
    max_depth=8,            # limit depth
    min_samples_split=100,  # avoid very small nodes
    random_state=42,
    class_weight="balanced",
    n_jobs=1                # single core (less overhead)
)

rf.fit(X_train_scaled, y_train)

# 8. Evaluation
y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]

print("=== Random Forest on Online Payments (100k rows) ===")
print(classification_report(y_test, y_pred_rf))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest (Online Payments)")
plt.tight_layout()
plt.savefig("../static/fraud_confusion_matrix.png")
plt.close()

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob_rf)
auc = roc_auc_score(y_test, y_prob_rf)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest (Online Payments)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("../static/fraud_roc_curve.png")
plt.close()
with open("../models/fraud_model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open("../models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("../models/type_encoder.pkl", "wb") as f:
    pickle.dump(type_encoder, f)

print("Final model, scaler, type encoder, and plots saved successfully!")
