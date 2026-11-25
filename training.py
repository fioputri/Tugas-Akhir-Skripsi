
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


# ==========================================
# 1. KONFIGURASI DATASET
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sales_raw.csv"
OUTPUT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)

# Jika nama kolom tidak terdeteksi otomatis,
# isi nama kolom sesuai CSV milikmu.
DATE_COLUMN = "DateTime"
PRODUCT_COLUMN = "DrugName"
QUANTITY_COLUMN = "Quantity"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ==========================================
# 2. MEMBACA DATASET
# ==========================================

def detect_column(df, manual_name, candidates, label):

    if manual_name is not None:
        if manual_name not in df.columns:
            raise ValueError(
                f"Kolom {manual_name} tidak ditemukan. "
                f"Kolom tersedia: {list(df.columns)}"
            )
        return manual_name

    normalized_columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.lower()

        if key in normalized_columns:
            return normalized_columns[key]

    raise ValueError(
        f"Kolom {label} tidak ditemukan.\n"
        f"Kolom tersedia: {list(df.columns)}\n"
        f"Silakan isi konfigurasi kolom secara manual."
    )


def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)

    print("Dataset berhasil dibaca.")
    print("Ukuran dataset:", df.shape)
    print("Kolom dataset:", list(df.columns))

    date_col = detect_column(
        df,
        DATE_COLUMN,
        ["Date", "TransactionDate", "SalesDate",
         "Tanggal", "Transaction_Date"],
        "tanggal"
    )

    product_col = detect_column(
        df,
        PRODUCT_COLUMN,
        ["ProductCode", "ProductID", "ItemCode",
         "DrugCode", "MedicineCode", "Product"],
        "produk"
    )

    quantity_col = detect_column(
        df,
        QUANTITY_COLUMN,
        ["Quantity", "Qty", "SalesQuantity",
         "QuantitySold", "UnitsSold", "Jumlah"],
        "kuantitas"
    )

    return df, date_col, product_col, quantity_col


# ==========================================
# 3. PREPROCESSING DAN AGREGASI DATA
# ==========================================

def preprocess_data(df, date_col, product_col, quantity_col):

    data = df[
        [date_col, product_col, quantity_col]
    ].copy()

    data.columns = [
        "date",
        "product_code",
        "quantity"
    ]

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    data["product_code"] = (
        data["product_code"]
        .astype("string")
        .str.strip()
    )

    data["quantity"] = pd.to_numeric(
        data["quantity"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["date", "product_code", "quantity"]
    )

    data["date"] = data["date"].dt.normalize()

    # Menggabungkan transaksi produk pada tanggal yang sama.
    daily_sales = (
        data.groupby(
            ["product_code", "date"],
            as_index=False
        )
        .agg(quantity=("quantity", "sum"))
    )

    daily_sales = daily_sales.sort_values(
        ["product_code", "date"]
    ).reset_index(drop=True)

    print("Preprocessing selesai.")
    print("Jumlah data harian:", len(daily_sales))

    daily_sales.to_csv(
        OUTPUT_DIR / "daily_sales_dataset.csv",
        index=False
    )

    return daily_sales


# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================

def create_features(data):

    data = data.copy()

    data = data.sort_values(
        ["product_code", "date"]
    ).reset_index(drop=True)

    grouped = data.groupby("product_code")["quantity"]

    # Fitur riwayat penjualan
    for lag in [1, 2, 7, 14, 28]:
        data[f"lag_{lag}"] = grouped.shift(lag)

    # Rata-rata penjualan sebelumnya
    for window in [7, 14, 28]:

        data[f"rolling_mean_{window}"] = (
            data.groupby("product_code")["quantity"]
            .transform(
                lambda series:
                series.shift(1)
                .rolling(window, min_periods=1)
                .mean()
            )
        )

    # Fitur waktu
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_month"] = data["date"].dt.day
    data["month"] = data["date"].dt.month
    data["quarter"] = data["date"].dt.quarter
    data["day_of_year"] = data["date"].dt.dayofyear

    feature_columns = [
        "product_code",
        "lag_1",
        "lag_2",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "day_of_week",
        "day_of_month",
        "month",
        "quarter",
        "day_of_year"
    ]

    model_data = data.dropna(
        subset=[
            "lag_1",
            "lag_2",
            "lag_7",
            "lag_14",
            "lag_28"
        ]
    ).copy()

    return model_data, feature_columns


# ==========================================
# 5. MEMBUAT PIPELINE MODEL
# ==========================================

def create_pipeline(model):

    categorical_features = ["product_code"]

    numerical_features = [
        "lag_1",
        "lag_2",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "day_of_week",
        "day_of_month",
        "month",
        "quarter",
        "day_of_year"
    ]

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "product",
                encoder,
                categorical_features
            ),
            (
                "numeric",
                "passthrough",
                numerical_features
            )
        ]
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("model", model)
    ])

    return pipeline


# ==========================================
# 6. EVALUASI MODEL
# ==========================================

def calculate_mape(actual, predicted):

    actual = np.array(actual)
    predicted = np.array(predicted)

    non_zero = actual != 0

    if non_zero.sum() == 0:
        return np.nan

    mape = np.mean(
        np.abs(
            (actual[non_zero] - predicted[non_zero])
            / actual[non_zero]
        )
    ) * 100

    return mape


# ==========================================
# 7. TRAINING RANDOM FOREST DAN XGBOOST
# ==========================================

def main():

    df, date_col, product_col, quantity_col = load_dataset()

    daily_sales = preprocess_data(
        df,
        date_col,
        product_col,
        quantity_col
    )

    model_data, feature_columns = create_features(
        daily_sales
    )

    if model_data.empty:
        raise ValueError(
            "Data untuk training kosong. "
            "Periksa jumlah riwayat penjualan setiap produk."
        )

    # Pembagian data berdasarkan urutan tanggal.
    unique_dates = sorted(model_data["date"].unique())

    if len(unique_dates) < 2:
        raise ValueError(
            "Tanggal unik tidak cukup untuk pembagian data."
        )

    split_index = int(
        len(unique_dates) * (1 - TEST_SIZE)
    )

    split_index = min(
        max(split_index, 1),
        len(unique_dates) - 1
    )

    cutoff_date = unique_dates[split_index]

    train_data = model_data[
        model_data["date"] < cutoff_date
    ]

    test_data = model_data[
        model_data["date"] >= cutoff_date
    ]

    X_train = train_data[feature_columns]
    y_train = train_data["quantity"]

    X_test = test_data[feature_columns]
    y_test = test_data["quantity"]

    print("\nJumlah data training:", len(train_data))
    print("Jumlah data testing:", len(test_data))

    # Model yang dibandingkan
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "XGBoost": XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    results = {}

    for model_name, estimator in models.items():

        print(f"\nTraining {model_name}...")

        pipeline = create_pipeline(estimator)

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        predictions = np.maximum(predictions, 0)

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        mape = calculate_mape(
            y_test,
            predictions
        )

        results[model_name] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "MAPE": (
                None if np.isnan(mape)
                else float(mape)
            )
        }

        print(f"{model_name}")
        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"MAPE : {mape:.4f}%")

    # Menyimpan hasil evaluasi
    results_df = pd.DataFrame([
        {
            "Model": name,
            "MAE": values["MAE"],
            "RMSE": values["RMSE"],
            "MAPE": values["MAPE"]
        }
        for name, values in results.items()
    ])

    results_df = results_df.sort_values(
        ["RMSE", "MAE"]
    )

    results_df.to_csv(
        OUTPUT_DIR / "evaluation_results.csv",
        index=False
    )

    # Memilih model dengan RMSE terendah
    best_model_name = results_df.iloc[0]["Model"]

    print("\nModel terbaik:", best_model_name)

    # Training ulang model terpilih dengan seluruh data
    best_model = create_pipeline(
        models[best_model_name]
    )

    best_model.fit(
        model_data[feature_columns],
        model_data["quantity"]
    )

    # Menyimpan model untuk website
    joblib.dump(
        best_model,
        OUTPUT_DIR / "best_model.joblib"
    )

    metadata = {
        "best_model": best_model_name,
        "feature_columns": feature_columns,
        "date_column": date_col,
        "product_column": product_col,
        "quantity_column": quantity_col,
        "last_data_date": str(
            daily_sales["date"].max().date()
        ),
        "selection_metric": "RMSE"
    }

    with open(
        OUTPUT_DIR / "model_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\nTraining selesai!")
    print("File tersimpan di folder artifacts.")


if __name__ == "__main__":
    main()
