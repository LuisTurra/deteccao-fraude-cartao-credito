import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from preprocess import carregar_dados, preprocessar, balancear

df, X, y = carregar_dados()
X, scaler = preprocessar(X)
X_res, y_res = balancear(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='auc'
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

# Salvar modelo
joblib.dump(model, "models/xgb_fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Modelo salvo!")