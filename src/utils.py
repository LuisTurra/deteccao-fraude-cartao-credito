import pandas as pd

def mostrar_metricas(y_test, y_pred, y_prob):
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Função pra carregar colunas (usada no app)
def get_feature_names():
    df = pd.read_csv("data/creditcard.csv")
    return df.drop('Class', axis=1).columns.tolist()