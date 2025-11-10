# modelling.py (versi fix)
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ======================
# 1. Load Dataset
# ======================
data = pd.read_csv("heart_disease_uci_preprocessing.csv")

# Pastikan kolom target ada
if "num" not in data.columns:
    raise ValueError("Kolom target 'num' tidak ditemukan dalam dataset!")

# ======================
# 2. Encoding untuk kolom kategorikal
# ======================
for col in data.columns:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

# Pastikan semua data numerik
print("✅ Semua kolom sudah numerik.")
print(data.dtypes)

# ======================
# 3. Split Data
# ======================
X = data.drop(columns=["num"])
y = data["num"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================
# 4. MLflow Autolog
# ======================
mlflow.set_experiment("Heart Disease Model - Basic")

with mlflow.start_run():
    mlflow.sklearn.autolog()  # otomatis log parameter & metrik

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Model selesai dilatih. Akurasi: {acc:.4f}")

print("📊 Lihat hasil tracking lokal dengan perintah: mlflow ui")
print("🌐 Akses di browser: http://127.0.0.1:5000")