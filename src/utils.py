import pandas as pd
import streamlit as st
import os
import requests
from io import StringIO
import kagglehub
from pathlib import Path

def mostrar_metricas(y_test, y_pred, y_prob):
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

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
        
       
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

    return df

def get_feature_names():
    df = load_data()
    return df.drop('Class', axis=1).columns.tolist()