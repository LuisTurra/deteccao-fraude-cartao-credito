import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os
import streamlit as st
import requests
from io import StringIO

from pathlib import Path

import gdown
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

def carregar_dados():
    df = load_data()
    X = df.drop('Class', axis=1)
    y = df['Class']
    return df, X, y

def preprocessar(X):
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    X['Time'] = scaler.fit_transform(X[['Time']])
    return X, scaler

def balancear(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res