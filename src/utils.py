import pandas as pd
import streamlit as st
import os
import requests
from io import StringIO

from pathlib import Path
import gdown

def mostrar_metricas(y_test, y_pred, y_prob):
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))



@st.cache_data
def load_data():
    HF_URL = "https://huggingface.co/datasets/luisturra/creditCardFraud/resolve/main/creditcard.csv"
    LOCAL_CACHE = "creditcard.csv"

    if not os.path.exists(LOCAL_CACHE):
        st.info("Baixando dataset (~150MB) do Hugging Face... aguarde 20-40 segundos")
        try:
            response = requests.get(HF_URL)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            df.to_csv(LOCAL_CACHE, index=False)
            st.success("Dataset baixado com sucesso!")
        except Exception as e:
            st.error(f"Erro no download: {e}")
            return pd.DataFrame()
    else:
        st.success("Dataset carregado do cache local!")
        df = pd.read_csv(LOCAL_CACHE)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
    
    return df

def get_feature_names():
    df = load_data()
    return df.drop('Class', axis=1).columns.tolist()