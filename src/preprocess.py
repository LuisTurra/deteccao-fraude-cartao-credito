import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os
import streamlit as st

from io import StringIO

from pathlib import Path

HF_REPO_ID = "luisturra/creditCardFraud" 
HF_FILE_NAME = "creditcard.csv"
@st.cache_data
def load_data():
   
    HF_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{HF_FILE_NAME}"
    

    LOCAL_CACHE_PATH = HF_FILE_NAME
    
    df = pd.DataFrame()

    if not os.path.exists(LOCAL_CACHE_PATH):
        st.info(f"Baixando dataset (~150MB) do Hugging Face Hub. Isto levará alguns segundos...")
        
        try:
           
            df = pd.read_csv(HF_URL, low_memory=False) 
            
           
            df.to_csv(LOCAL_CACHE_PATH, index=False)
            st.success("Dataset baixado e salvo no cache com sucesso!")

        except Exception as e:
            st.error(f"ERRO DE DOWNLOAD: Falha ao baixar o arquivo do Hugging Face. Detalhes: {e}")
            return pd.DataFrame() 
    
    else:
        st.success("Dataset carregado do cache local!")
       
        df = pd.read_csv(LOCAL_CACHE_PATH)
        
       
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