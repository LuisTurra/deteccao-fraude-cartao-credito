import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def carregar_dados():
    df = pd.read_csv("data/creditcard.csv")
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