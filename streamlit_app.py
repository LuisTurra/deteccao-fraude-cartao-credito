import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from src.utils import get_feature_names

st.set_page_config(page_title="Detecção de Fraude - Luis Turra", layout="wide")
st.title("Detecção de Fraude em Cartão de Crédito")

@st.cache_data
def load_data():
    return pd.read_csv("data/creditcard.csv")

df = load_data()
model = joblib.load("models/xgb_fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")
X_columns = df.drop('Class', axis=1).columns

tab1, tab2, tab3 = st.tabs(["Dashboard", "Testar Transação", "EDA"])

with tab1:
    st.header("Dashboard Geral")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de transações", f"{len(df):,}")
    col2.metric("Fraudes", f"{df['Class'].sum():,}")
    col3.metric("% de fraude", f"{df['Class'].mean()*100:.4f}%")

    fig = px.pie(values=[len(df)-df['Class'].sum(), df['Class'].sum()],
                 names=['Normal', 'Fraude'], title="Distribuição")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Teste em tempo real")
    valor = st.number_input("Valor (R$)", 0.0, 30000.0, 85.0)
    tempo = st.number_input("Tempo (segundos)", 0, 172792, 50000)

    if st.button("Verificar"):
        features = np.zeros((1, 30))
        features[0, -2] = tempo
        features[0, -1] = valor
        features[0, -2] = scaler.transform([[tempo]])[0][0]
        features[0, -1] = scaler.transform([[valor]])[0][0]

        prob = model.predict_proba(features)[0][1]
        pred = model.predict(features)[0]

        st.metric("Probabilidade de fraude", f"{prob:.2%}")
        if pred == 1:
            st.error("FRAUDE DETECTADA!")
        else:
            st.success("Transação SEGURA")

        # SHAP corrigido
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)
        plt.figure(figsize=(12, 4))
        shap.force_plot(explainer.expected_value, shap_values[0], features[0],
                        feature_names=X_columns, matplotlib=True, show=False)
        st.pyplot(plt)
        plt.clf()

with tab3:
    st.header("Análise Exploratória")
    st.markdown("Notebook completo:")
    st.markdown("GitHub: [notebooks/01_EDA_Analise_Exploratoria.ipynb](https://github.com/luisturra/credit-card-fraud-detection/blob/main/notebooks/01_EDA_Analise_Exploratoria.ipynb)")