
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ==========================================
# 1. KONFIGURASI WEBSITE
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
DAILY_DATA_PATH = ARTIFACT_DIR / "daily_sales_dataset.csv"
EVALUATION_PATH = ARTIFACT_DIR / "evaluation_results.csv"


# ==========================================
# 2. PENGATURAN HALAMAN
# ==========================================

st.set_page_config(
    page_title="Prediksi Penjualan Apotek",
    page_icon="📊",
    layout="wide"
)

st.title("Sistem Prediksi Penjualan Apotek")

st.write(
    "Aplikasi ini digunakan untuk menampilkan hasil "
    "prediksi penjualan produk apotek dan indikator "
    "rekomendasi persediaan berdasarkan model Machine Learning."
)


# ==========================================
# 3. MEMUAT MODEL DAN DATA
# ==========================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():

    if not METADATA_PATH.exists():
        return None

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


@st.cache_data
def load_daily_data():

    if not DAILY_DATA_PATH.exists():
        return None

    data = pd.read_csv(DAILY_DATA_PATH)

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    data["product_code"] = (
        data["product_code"].astype(str)
    )

    data["quantity"] = pd.to_numeric(
        data["quantity"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["date", "product_code", "quantity"]
    )

    return data


@st.cache_data
def load_evaluation():

    if not EVALUATION_PATH.exists():
        return None

    return pd.read_csv(EVALUATION_PATH)


model = load_model()
metadata = load_metadata()
daily_data = load_daily_data()
evaluation = load_evaluation()


# ==========================================
# 4. MEMERIKSA KETERSEDIAAN MODEL
# ==========================================

if (
    model is None
    or metadata is None
    or daily_data is None
):

    st.error(
        "Model belum tersedia. "
        "Silakan jalankan training.py terlebih dahulu."
    )

    st.stop()


# ==========================================
# 5. INFORMASI MODEL
# ==========================================

st.subheader("Informasi Model")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Model Terpilih",
    metadata["best_model"]
)

col2.metric(
    "Jumlah Produk",
    daily_data["product_code"].nunique()
)

col3.metric(
    "Jumlah Data",
    f"{len(daily_data):,}"
)

st.divider()


# ==========================================
# 6. FUNGSI MEMBUAT FITUR PREDIKSI
# ==========================================

def create_prediction_features(product_data):

    product_data = product_data.sort_values(
        "date"
    )

    quantities = (
        product_data["quantity"]
        .astype(float)
        .tolist()
    )

    if len(quantities) < 28:

        raise ValueError(
            "Riwayat penjualan produk belum cukup "
            "untuk membuat prediksi."
        )

    last_date = product_data["date"].max()

    next_date = (
        pd.Timestamp(last_date)
        + pd.Timedelta(days=1)
    )

    features = {
        "product_code": str(
            product_data["product_code"].iloc[0]
        ),

        "lag_1": quantities[-1],
        "lag_2": quantities[-2],
        "lag_7": quantities[-7],
        "lag_14": quantities[-14],
        "lag_28": quantities[-28],

        "rolling_mean_7": np.mean(
            quantities[-7:]
        ),

        "rolling_mean_14": np.mean(
            quantities[-14:]
        ),

        "rolling_mean_28": np.mean(
            quantities[-28:]
        ),

        "day_of_week": next_date.dayofweek,
        "day_of_month": next_date.day,
        "month": next_date.month,
        "quarter": next_date.quarter,
        "day_of_year": next_date.dayofyear
    }

    return pd.DataFrame([features]), next_date


# ==========================================
# 7. MENU WEBSITE
# ==========================================

menu = st.sidebar.selectbox(
    "Menu Navigasi",
    [
        "Dashboard",
        "Prediksi Penjualan",
        "Evaluasi Model",
        "Data Penjualan"
    ]
)


# ==========================================
# 8. DASHBOARD
# ==========================================

if menu == "Dashboard":

    st.subheader("Dashboard")

    st.write(
        "Dashboard menampilkan informasi umum "
        "mengenai dataset dan model prediksi."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Produk",
            daily_data["product_code"].nunique()
        )

    with col2:
        st.metric(
            "Total Observasi",
            f"{len(daily_data):,}"
        )

    st.write("Model yang digunakan:")
    st.info(metadata["best_model"])

    st.write(
        "Tanggal terakhir dalam dataset:",
        daily_data["date"].max().strftime("%d-%m-%Y")
    )


# ==========================================
# 9. PREDIKSI PENJUALAN
# ==========================================

elif menu == "Prediksi Penjualan":

    st.subheader("Prediksi Penjualan Produk")

    product_list = sorted(
        daily_data["product_code"]
        .astype(str)
        .unique()
        .tolist()
    )

    selected_product = st.selectbox(
        "Pilih Produk",
        product_list
    )

    product_data = daily_data[
        daily_data["product_code"].astype(str)
        == selected_product
    ].sort_values("date")

    st.write(
        "Jumlah riwayat penjualan:",
        len(product_data)
    )

    if st.button(
        "Prediksi Penjualan",
        type="primary"
    ):

        try:

            features, prediction_date = (
                create_prediction_features(
                    product_data
                )
            )

            prediction = float(
                model.predict(features)[0]
            )

            prediction = max(
                prediction,
                0
            )

            st.session_state["prediction"] = {
                "product": selected_product,
                "date": prediction_date,
                "quantity": prediction
            }

        except ValueError as error:

            st.error(str(error))


    # Menampilkan hasil prediksi

    result = st.session_state.get(
        "prediction"
    )

    if (
        result is not None
        and result["product"] == selected_product
    ):

        st.success("Prediksi berhasil dilakukan.")

        col1, col2 = st.columns(2)

        col1.metric(
            "Produk",
            result["product"]
        )

        col2.metric(
            "Prediksi Penjualan",
            f"{result['quantity']:.2f} unit"
        )

        st.write(
            "Tanggal prediksi:",
            result["date"].strftime("%d-%m-%Y")
        )

        st.divider()

        # ==================================
        # REKOMENDASI PERSEDIAAN
        # ==================================

        st.subheader("Rekomendasi Persediaan")

        st.write(
            "Masukkan jumlah stok produk yang tersedia "
            "saat ini untuk membandingkannya dengan "
            "hasil prediksi penjualan."
        )

        current_stock = st.number_input(
            "Jumlah Stok Tersedia",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

        if st.button(
            "Hitung Rekomendasi Persediaan"
        ):

            predicted_sales = result["quantity"]

            if current_stock < predicted_sales:

                difference = (
                    predicted_sales - current_stock
                )

                st.warning(
                    "Stok tersedia lebih rendah "
                    "dibandingkan perkiraan penjualan."
                )

                st.write(
                    f"Selisih stok: {difference:.2f} unit"
                )

                st.write(
                    "Indikator: stok perlu diperiksa "
                    "untuk mengantisipasi kebutuhan produk."
                )

            else:

                difference = (
                    current_stock - predicted_sales
                )

                st.success(
                    "Stok tersedia mencukupi "
                    "dibandingkan perkiraan penjualan satu hari."
                )

                st.write(
                    f"Selisih stok: {difference:.2f} unit"
                )

                st.write(
                    "Indikator: stok dapat dipantau "
                    "berdasarkan perkiraan penjualan."
                )

        st.caption(
            "Rekomendasi ini merupakan indikator sederhana "
            "berdasarkan prediksi satu hari dan input stok. "
            "Belum memperhitungkan lead time, safety stock, "
            "atau stok yang sedang dalam proses pemesanan."
        )


# ==========================================
# 10. EVALUASI MODEL
# ==========================================

elif menu == "Evaluasi Model":

    st.subheader("Hasil Evaluasi Model")

    st.write(
        "Perbandingan kinerja algoritma "
        "Random Forest dan XGBoost."
    )

    if evaluation is not None:

        st.dataframe(
            evaluation,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            f"Model terpilih berdasarkan evaluasi RMSE: "
            f"{metadata['best_model']}"
        )

    else:

        st.warning(
            "Data evaluasi belum tersedia."
        )


# ==========================================
# 11. DATA PENJUALAN
# ==========================================

elif menu == "Data Penjualan":

    st.subheader("Data Penjualan")

    st.dataframe(
        daily_data.sort_values(
            "date",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Grafik Penjualan")

    selected_chart_product = st.selectbox(
        "Pilih Produk untuk Grafik",
        sorted(
            daily_data["product_code"]
            .astype(str)
            .unique()
            .tolist()
        ),
        key="chart_product"
    )

    chart_data = daily_data[
        daily_data["product_code"].astype(str)
        == selected_chart_product
    ].sort_values("date")

    st.line_chart(
        chart_data.set_index("date")["quantity"]
    )


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "Sistem Prediksi Penjualan dan Rekomendasi "
    "Persediaan Produk Apotek."
)
