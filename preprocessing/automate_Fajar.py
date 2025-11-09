import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(input_path, output_path):
    """
    Melakukan preprocessing otomatis pada dataset heart_disease_uci.csv
    sesuai dengan tahapan manual pada Eksperimen_Fajar.ipynb
    """

    # ======================================================
    # 1. Load Dataset
    # ======================================================
    df = pd.read_csv(input_path)
    print("✅ Dataset berhasil dimuat.")
    print(f"Jumlah data awal: {df.shape[0]} baris, {df.shape[1]} kolom\n")

    # ======================================================
    # 2. Menghapus Kolom Tidak Relevan
    # ======================================================
    df = df.drop(columns=['id', 'dataset'], errors='ignore')
    print("🧹 Kolom tidak relevan ('id', 'dataset') telah dihapus.\n")

    # ======================================================
    # 3. Menangani Missing Values
    # ======================================================
    # Numerik → isi dengan median
    num_cols = ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Kategorikal → isi dengan modus
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    print("🔧 Missing values telah ditangani.\n")

    # ======================================================
    # 4. Menghapus Data Duplikat
    # ======================================================
    before = df.shape[0]
    df = df.drop_duplicates()
    print(f"🧩 Dihapus {before - df.shape[0]} data duplikat.\n")

    # ======================================================
    # 5. Encoding Fitur Kategorikal
    # ======================================================
    # Label Encoding untuk kolom ordinal
    le = LabelEncoder()
    if 'slope' in df.columns:
        df['slope'] = le.fit_transform(df['slope'])

    # One-Hot Encoding untuk kolom kategorikal nominal
    one_hot_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'thal']
    df = pd.get_dummies(df, columns=[col for col in one_hot_cols if col in df.columns], drop_first=True)

    print("🔡 Encoding fitur kategorikal selesai.\n")

    # ======================================================
    # 6. Deteksi dan Penanganan Outlier
    # ======================================================
    num_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    existing_num_features = [col for col in num_features if col in df.columns]
    
    Q1 = df[existing_num_features].quantile(0.25)
    Q3 = df[existing_num_features].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[existing_num_features] < (Q1 - 1.5 * IQR)) | (df[existing_num_features] > (Q3 + 1.5 * IQR))).any(axis=1)]

    print("🚫 Outlier telah dideteksi dan dihapus.\n")

    # ======================================================
    # 7. Normalisasi / Standarisasi Fitur Numerik
    # ======================================================
    scaler = StandardScaler()
    df[existing_num_features] = scaler.fit_transform(df[existing_num_features])
    print("📏 Fitur numerik telah dinormalisasi.\n")

    # ======================================================
    # 8. Binning (Pengelompokan Umur)
    # ======================================================
    if 'age' in df.columns:
        bins = [0, 30, 40, 50, 60, 70, 80, 100]
        labels = ['<30', '30-40', '40-50', '50-60', '60-70', '70-80', '>80']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
        print("📊 Binning usia telah ditambahkan.\n")

    # ======================================================
    # 9. Memisahkan Fitur dan Target
    # ======================================================
    if 'num' in df.columns:
        X = df.drop(columns=['num'])
        y = df['num']
        print("✅ Fitur (X) dan target (y) telah dipisahkan.\n")
    else:
        X, y = df, None
        print("⚠️ Kolom target 'num' tidak ditemukan.\n")

    # ======================================================
    # 10. Simpan Dataset Hasil Preprocessing
    # ======================================================
    df.to_csv(output_path, index=False)
    print(f"💾 Dataset hasil preprocessing disimpan ke: {output_path}")
    print(f"Jumlah data akhir: {df.shape[0]} baris, {df.shape[1]} kolom\n")

    return df


# ======================================================
# Eksekusi Langsung Jika File Dijalankan
# ======================================================
if __name__ == "__main__":
    input_path = "../heart_disease_uci_raw/heart_disease_uci.csv"
    output_path = "../preprocessing/heart_disease_uci_preprocessing.csv"
    df_cleaned = preprocess_data(input_path, output_path)
    print("\nPreview dataset hasil preprocessing:")
    print(df_cleaned.head())