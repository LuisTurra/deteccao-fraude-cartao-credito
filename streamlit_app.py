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
    st.header("Resumo e KPIs Chave")
    
    # -------------------------------------------------------------
    # 1. Linha de Métricas (KPIs)
    # -------------------------------------------------------------
    total_transacoes = len(df)
    total_fraudes = df['Class'].sum()
    porcentagem_fraude = df['Class'].mean() * 100
    valor_fraude = df[df['Class'] == 1]['Amount'].sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total de Transações", f"{total_transacoes:,}")
    col2.metric("Fraudes Registradas (Count)", f"{total_fraudes:,}")
    col3.metric("% de Fraude (Taxa)", f"{porcentagem_fraude:.4f}%")
    col4.metric("**Valor Total em Risco (Fraude)**", f"R$ {valor_fraude:,.2f}")

    st.markdown("---")

    # -------------------------------------------------------------
    # 2. Primeira Linha de Gráficos (Classes e Amount)
    # -------------------------------------------------------------
    st.subheader("Análise de Distribuição Geral")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("##### Distribuição de Classes")
        df_classes = pd.DataFrame({'Tipo': ['Normal', 'Fraude'], 'Contagem': [total_transacoes - total_fraudes, total_fraudes]})
        
        fig_pie = px.pie(
            df_classes, 
            values='Contagem', 
            names='Tipo', 
            title="Distribuição de Transações (Normal vs. Fraude)",
            color='Tipo',
            color_discrete_map={'Normal':'#1f77b4', 'Fraude':'#d62728'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with colB:
        st.markdown("##### Distribuição de Valor (`Amount`) por Classe")
        
        fig_box_amount = px.box(
            df, 
            y="Amount", 
            color="Class", 
            title="Box Plot de 'Amount'", 
            labels={'Class': 'Classe (0=Normal, 1=Fraude)'}, 
            color_discrete_map={0: '#1f77b4', 1: '#d62728'}
        )
        fig_box_amount.update_layout(yaxis_title="Valor da Transação (R$)")
        st.plotly_chart(fig_box_amount, use_container_width=True)

    st.markdown("---")

    # -------------------------------------------------------------
    # 3. Segunda Linha de Gráficos (Análise de Características Vⁿ - Fraude Predictors)
    # -------------------------------------------------------------
    st.subheader("Análise de Características PCA ($V_n$) - Preditoras de Fraude")
    st.markdown("Use o seletor para visualizar o poder de separação de cada variável $V_n$ anonimizada.")

    # 3.1. Preparação dos dados para seleção
    v_cols = [col for col in df.columns if col.startswith('V')]
    corr_data = df[v_cols + ['Class']].corr()
    # Ordena pelo valor absoluto da correlação para identificar as mais importantes
    corr_class = corr_data['Class'].drop('Class').sort_values(key=abs, ascending=False) 

    # 3.2. Criação das colunas para os gráficos da Vⁿ
    colC, colD = st.columns([1, 2])

    with colC:
        st.markdown("##### 💡 Top Features por Correlação com Fraude")
        
        # Mostra as Top Features em uma tabela simplificada
        top_corr_df = corr_class.head(6).reset_index().rename(columns={'index': 'Feature', 'Class': 'Correlação'})
        top_corr_df['Correlação'] = top_corr_df['Correlação'].round(4)
        st.dataframe(top_corr_df, hide_index=True)
        
        # Seletor para Box Plot
        selected_feature = st.selectbox(
            "Selecione uma Feature para ver a distribuição:",
            options=corr_class.index.tolist(),
            index=0 # Inicia na feature mais correlacionada
        )

    with colD:
        st.markdown(f"##### Box Plot de Distribuição: **{selected_feature}**")
        
        # Box Plot da Feature selecionada
        fig_dist_vn = px.box(
            df, 
            y=selected_feature, 
            color="Class", 
            title=f"Distribuição de {selected_feature} por Classe (0=Normal, 1=Fraude)",
            labels={'Class': 'Classe'},
            color_discrete_map={0: '#1f77b4', 1: '#d62728'}
        )
        st.plotly_chart(fig_dist_vn, use_container_width=True)

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