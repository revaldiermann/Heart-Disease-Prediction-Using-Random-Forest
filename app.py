import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------
# CONFIG
# ------------------

st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="❤️",
    layout="wide"
)

# ------------------
# LOAD DATA
# ------------------

df = pd.read_csv("data/heart.csv")

model = joblib.load(
    "model/random_forest.pkl"
)

# ------------------
# HEADER
# ------------------

st.title("❤️ Prediksi Risiko Penyakit Jantung")

st.markdown("""
### UAS Data Science

**Metode : Random Forest**

Aplikasi ini digunakan untuk memprediksi risiko penyakit jantung berdasarkan data kesehatan pasien.
""")

# ------------------
# SIDEBAR
# ------------------

st.sidebar.header("Input Data Pasien")

age = st.sidebar.slider("Usia",20,100,50)

sex = st.sidebar.selectbox(
    "Jenis Kelamin",
    [0,1],
    format_func=lambda x:
    "Perempuan" if x == 0 else "Laki-laki"
)

cp = st.sidebar.selectbox(
    "Chest Pain Type",
    [0,1,2,3]
)

trestbps = st.sidebar.slider(
    "Tekanan Darah",
    80,
    220,
    120
)

chol = st.sidebar.slider(
    "Kolesterol",
    100,
    600,
    200
)

fbs = st.sidebar.selectbox(
    "Fasting Blood Sugar",
    [0,1]
)

restecg = st.sidebar.selectbox(
    "Rest ECG",
    [0,1,2]
)

thalach = st.sidebar.slider(
    "Detak Jantung Maksimum",
    60,
    220,
    150
)

exang = st.sidebar.selectbox(
    "Exercise Angina",
    [0,1]
)

oldpeak = st.sidebar.number_input(
    "Oldpeak",
    value=1.0
)

slope = st.sidebar.selectbox(
    "Slope",
    [0,1,2]
)

ca = st.sidebar.selectbox(
    "Jumlah Pembuluh Darah",
    [0,1,2,3,4]
)

thal = st.sidebar.selectbox(
    "Thal",
    [0,1,2,3]
)

# ------------------
# DATA INPUT
# ------------------

input_data = pd.DataFrame({
    'age':[age],
    'sex':[sex],
    'cp':[cp],
    'trestbps':[trestbps],
    'chol':[chol],
    'fbs':[fbs],
    'restecg':[restecg],
    'thalach':[thalach],
    'exang':[exang],
    'oldpeak':[oldpeak],
    'slope':[slope],
    'ca':[ca],
    'thal':[thal]
})

# ------------------
# PREDIKSI
# ------------------

col1,col2 = st.columns(2)

with col1:

    st.subheader("Data Pasien")

    st.dataframe(
        input_data,
        use_container_width=True
    )

    if st.button("🔍 Prediksi Risiko"):

        hasil = model.predict(
            input_data
        )

        if hasil[0] == 1:

            st.error(
                "⚠️ Pasien Berisiko Penyakit Jantung"
            )

        else:

            st.success(
                "✅ Pasien Tidak Berisiko Penyakit Jantung"
            )

with col2:

    st.subheader("Distribusi Target")

    fig, ax = plt.subplots()

    sns.countplot(
        x='target',
        data=df,
        ax=ax
    )

    st.pyplot(fig)

# ------------------
# HEATMAP
# ------------------

st.subheader("Heatmap Korelasi")

fig2, ax2 = plt.subplots(
    figsize=(10,6)
)

sns.heatmap(
    df.corr(),
    cmap='coolwarm',
    ax=ax2
)

st.pyplot(fig2)

# ------------------
# DATASET
# ------------------

st.subheader("Dataset Heart Disease")

st.dataframe(
    df,
    use_container_width=True
)
