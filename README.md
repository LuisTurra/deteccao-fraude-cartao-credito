# Detecção de Fraude em Cartão de Crédito

**XGBoost + SMOTE → ROC-AUC 0.9999**  
**Dashboard interativo (Streamlit + Plotly)**  


🔗 **Demo ao vivo**: https://luisturra-deteccao-fraude-cartao-credito-streamlit-app-99a0gz.streamlit.app/  
🔗 **Repositório**: https://github.com/LuisTurra/deteccao-fraude-cartao-credito  
🔗 **Notebook EDA**: notebooks/analise.ipynb

## Por que esse projeto?
- Fraude é o maior medo de Nubank, PicPay, C6, Mercado Pago, Stone...  
-  **XGBoost + SMOTE + ROC-AUC + deploy**  


## Funcionalidades
- Teste de transação em tempo real  
- Dashboard interativo
- Explicabilidade com SHAP  
- Análise exploratória completa

## Como rodar local
```bash
pip install -r requirements.txt
streamlit run app.py
