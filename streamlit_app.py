import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from src.utils import get_feature_names
import os
from pathlib import Path

st.set_page_config(page_title="Detecção de Fraude - Luis Turra", layout="wide")
st.title("Detecção de Fraude em Cartão de Crédito")
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

df = load_data()
model = joblib.load("models/xgb_fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")
X_columns = df.drop('Class', axis=1).columns

tab1, tab2 = st.tabs([ "Testar Transação", "EDA"])


with tab1:
    st.header("Teste em Tempo Real - Simulador de Transação")
    st.markdown("**Preencha os dados como se fosse uma compra real.** O modelo analisa em segundos.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dados da Transação")
        valor = st.number_input("Valor da compra (R$)", min_value=0.01, max_value=30000.0, value=89.90, step=0.01, format="%.2f")
        
        
        tempo = st.slider("Horário da transação", 0, 172792, 95000, 
                         help="Segundos desde a primeira transação do dia")
        
        
        np.random.seed(int(tempo) % 100)
        realistic_features = []
        for col in X_columns:
            if col in ['Time', 'Amount']:
                continue
            mean_val = df[col].mean()
            std_val = df[col].std()
            noise = np.random.normal(0, 0.3)
            val = mean_val + noise * std_val
            realistic_features.append(val)
        
        
        st.markdown("**Características PCA geradas automaticamente:**")
        v_df = pd.DataFrame({
            'Variável': X_columns[:-2],
            'Valor': [f"{x:.3f}" for x in realistic_features]
        })
        st.dataframe(v_df, use_container_width=True, hide_index=True)

        features = np.array([realistic_features[:28] + [tempo, valor]])

    with col2:
        st.subheader("Contexto da Transação (Opcional)")
        tipo_compra = st.selectbox("Tipo de compra", ["Online", "Loja Física", "App", "Recorrência"])
        dispositivo = st.selectbox("Dispositivo", ["Celular", "Computador", "Tablet", "POS"])
        pais = st.selectbox("País", ["Brasil", "Argentina", "Colômbia", "Outros"])
        nova_conta = st.checkbox("Primeira compra deste cartão?", value=False)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("VERIFICAR FRAUDE AGORA", type="primary", use_container_width=True):
        st.markdown("<hr>", unsafe_allow_html=True)

       
        features_scaled = features.copy()
        features_scaled[0, -2] = scaler.transform([[tempo]])[0][0]
        features_scaled[0, -1] = scaler.transform([[valor]])[0][0]

        
        prob = model.predict_proba(features_scaled)[0][1]
        pred = model.predict(features_scaled)[0]

        
        col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
        with col_res2:
            if pred == 1:
                st.error(f"FRAUDE DETECTADA!")
                st.warning(f"Probabilidade: **{prob:.2%}**")
                st.error("Transação BLOQUEADA automaticamente.")
            else:
                st.success(f"Transação APROVADA")
                if prob > 0.3:
                    st.warning(f"Probabilidade de fraude: **{prob:.2%}** → Monitoramento ativado")
                else:
                    st.info(f"Probabilidade de fraude: **{prob:.2%}** → Baixo risco")

        st.markdown("---")

        
        col_shap1, col_shap2 = st.columns([1, 10])
        with col_shap1:
            st.markdown("**i**")
        with col_shap2:
            st.caption("**O que é SHAP?** → Mostra quanto **cada variável contribuiu** para a decisão do modelo. Vermelho = aumentou o risco de fraude. Azul = diminuiu.")

        st.subheader("Por que o modelo decidiu isso?")

        with st.expander("O que é SHAP? (clique pra entender)", expanded=False):
            st.markdown("""
            **SHAP** = **SHapley Additive exPlanations**  
            É o jeito mais justo de descobrir **quem realmente influenciou** a decisão do modelo.

            **Como funciona no seu app:**
            - O modelo começa com uma **probabilidade base** (ex: 5%)
            - Cada variável (V14, Amount, etc.) **adiciona ou subtrai** uma porcentagem
            - O SHAP mostra **exatamente quanto cada uma mudou** o resultado final

            **Exemplo real:**
            ```
            Probabilidade inicial = 3%
            + V14 muito baixo      → +45% (principal culpado)
            + Valor alto           → +12%
            - V3 normal            → -18% (ajudou a defender)
            = Probabilidade final = 42% → FRAUDE!
            ```

            **No gráfico:**
            - **Vermelho** = aumentou o risco de fraude
            - **Verde** = diminuiu o risco
            - O tamanho da barra = força do impacto

            
            """)
            
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_scaled)
        shap_vals = np.array(shap_values[0], dtype=float).ravel()
        formatted_features = [f"{x:.2f}" for x in features_scaled[0]]

        
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        plt.figure(figsize=(20, 6))
        shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_vals.round(4),
            features=formatted_features,
            feature_names=X_columns,
            matplotlib=True,
            show=False,
            contribution_threshold=0.01
        )
        plt.title("Explicação SHAP - Impacto de cada variável", fontsize=18, pad=40)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()
        plt.close('all')

       
        col_tab1, col_tab2 = st.columns([1, 10])
        with col_tab1:
            st.markdown("**i**")
        with col_tab2:
            st.caption("**Variável** = nome (ex: V14) | **Valor** = valor usado na transação | **Impacto SHAP** = quanto mudou a probabilidade (+ = aumentou risco) | **Direção** = efeito final")

        st.markdown("**Top 10 variáveis que mais influenciaram:**")
        
        impact_df = pd.DataFrame({
            'Variável': X_columns,
            'Valor': formatted_features,
            'Impacto SHAP': shap_vals.round(4)
        })
        impact_df['Impacto Absoluto'] = impact_df['Impacto SHAP'].abs()
        impact_df['Direção'] = impact_df['Impacto SHAP'].apply(lambda x: "Aumentou risco" if x > 0 else "Reduziu risco")
        impact_df = impact_df.sort_values(by='Impacto Absoluto', ascending=False).head(10)

        def highlight_row(row):
            color = "#f1092c" if row['Direção'] == 'Aumentou risco' else "#0919f7"
            return [f'background-color: {color}'] * len(row)

        styled = impact_df[['Variável', 'Valor', 'Impacto SHAP', 'Direção']].style\
            .apply(highlight_row, axis=1)\
            .format({'Impacto SHAP': '{:+.4f}'})

        st.dataframe(styled, use_container_width=True, hide_index=True)

        top = impact_df.iloc[0]
        st.markdown(f"**Resumo:** **{top['Variável']}** = {top['Valor']} → **{top['Direção'].lower()}** o risco (impacto {top['Impacto SHAP']:+.4f}).")

       
        st.subheader("Ação Recomendada")
        if pred == 1:
            st.error("Bloquear + 3D Secure")
        elif prob > 0.7:
            st.warning("SMS / App")
        elif prob > 0.4:
            st.info("Monitorar 24h")
        else:
            st.success("Liberar")

        relatorio = f"""RELATÓRIO DE FRAUDE\nValor: R$ {valor:,.2f}\nProb: {prob:.2%}\nTop: {top['Variável']}"""
        st.download_button("Baixar Relatório", relatorio, f"fraude_{int(tempo)}.txt")
with tab2:
    st.header("Análise Exploratória de Dados (EDA)")
    

    st.subheader("Resumo do Dataset")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Transações", f"{len(df):,}")
    col2.metric("Fraudes Detectadas", f"{df['Class'].sum():,}")
    col3.metric("Taxa de Fraude", f"{df['Class'].mean()*100:.4f}%")
    col4.metric("Período (horas)", f"{(df['Time'].max()/3600):.1f}h")

    st.subheader("Distribuição Temporal das Transações")
    df_temp = df.copy()
    df_temp['Hora do Dia'] = (df_temp['Time'] % 86400) // 3600  # 86400 = 24h
    df_temp['Tipo'] = df_temp['Class'].map({0: 'Normal', 1: 'Fraude'})

    fig_time = px.histogram(
        df_temp, x='Hora do Dia', color='Tipo',
        nbins=24, title="Transações por Hora do Dia",
        labels={'Hora do Dia': 'Hora', 'count': 'Nº de Transações'},
        color_discrete_map={'Normal': '#1f77b4', 'Fraude': '#d62728'},
        barmode='overlay', opacity=0.7
    )
    fig_time.update_layout(
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
        yaxis_title="Quantidade",
        legend_title="Tipo"
    )
    st.plotly_chart(fig_time, use_container_width=True)

    st.caption("Observação: Fraudes ocorrem mais em horários de **madrugada** (0h-6h) — padrão clássico de ataque!")

    st.subheader("Top 10 Variáveis Mais Importantes (XGBoost)")
    importances = model.feature_importances_
    feat_importance = pd.DataFrame({
        'Feature': X_columns,
        'Importância': importances
    }).sort_values(by='Importância', ascending=False).head(10)

    fig_imp = px.bar(
        feat_importance, x='Importância', y='Feature',
        orientation='h', title="Feature Importance (Gain)",
        color='Importância', color_continuous_scale='Reds'
    )
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("Correlação das Variáveis com Fraude")
    corr = df.corr()['Class'].drop('Class').sort_values(key=abs, ascending=False)
    corr_df = corr.reset_index().rename(columns={'index': 'Feature', 'Class': 'Correlação'})

    fig_corr = px.bar(
        corr_df.head(10), x='Correlação', y='Feature',
        orientation='h', title="Top 10 Correlações com Fraude",
        color='Correlação', color_continuous_scale='RdBu'
    )
    fig_corr.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Distribuição das Principais Variáveis (Normal vs Fraude)")
    top_features = feat_importance['Feature'].head(6).tolist()
    selected_feat = st.selectbox("Selecione uma variável para ver o Boxplot:", top_features)

    fig_box = px.box(
        df, y=selected_feat, color='Class',
        title=f"Distribuição de {selected_feat} por Classe",
        labels={'Class': 'Classe (0=Normal, 1=Fraude)'},
        color_discrete_map={0: '#1f77b4', 1: '#d62728'}
    )
    st.plotly_chart(fig_box, use_container_width=True)

   
    