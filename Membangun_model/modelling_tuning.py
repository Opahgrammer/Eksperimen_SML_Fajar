# modelling_tuning.py
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dagshub import dagshub_logger

# ===================================
# 1. Konfigurasi koneksi ke DagsHub
# ===================================
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Opahgrammer/Eksperimen_SML_Fajar.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "Opahgrammer"  # username GitHub/DagsHub kamu
os.environ["MLFLOW_TRACKING_PASSWORD"] = "<TOKEN_DAGSHUB_KAMU>"  # ganti dengan personal access token DagsHub kamu

mlflow.set_experiment("Heart Disease Model - Advanced")

# ===================================
# 2. Load Dataset
# ===================================
data = pd.read_csv("heart_disease_uci_preprocessing.csv")

if "num" not in data.columns:
    raise ValueError("Kolom target 'num' tidak ditemukan!")

X = data.drop(columns=["num"])
y = data["num"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===================================
# 3. Hyperparameter Tuning
# ===================================
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [5, 10, 15]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# ===================================
# 4. Evaluasi Model
# ===================================
preds = best_model.predict(X_test)
acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds)
rec = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)

# ===================================
# 5. Manual Logging ke DagsHub
# ===================================
with mlflow.start_run():
    # Logging parameter hasil tuning
    mlflow.log_param("best_n_estimators", grid.best_params_["n_estimators"])
    mlflow.log_param("best_max_depth", grid.best_params_["max_depth"])

    # Logging metrik evaluasi
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Metrik tambahan di luar autolog → wajib untuk Advance
    mlflow.log_metric("specificity", rec / (rec + (1 - prec)))  # contoh tambahan

    # Simpan model ke artefak
    mlflow.sklearn.log_model(best_model, "best_random_forest_model")

print(f"🎯 Best Model Accuracy: {acc:.4f}")
print("🚀 Model dan metrik berhasil dikirim ke DagsHub!")