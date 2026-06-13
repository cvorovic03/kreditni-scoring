import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle

print("Treniranje XGBoost modela...")

# Generisanje podataka
np.random.seed(42)
broj_kompanija = 500

X1 = np.random.uniform(0, 0.5, broj_kompanija)
X2 = np.random.uniform(0, 0.4, broj_kompanija)
X3 = np.random.uniform(0, 0.4, broj_kompanija)
X4 = np.random.uniform(0, 3, broj_kompanija)
X5 = np.random.uniform(0, 2, broj_kompanija)

altman_z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

y = np.zeros(broj_kompanija)
for i in range(broj_kompanija):
    if altman_z[i] > 2.9:
        y[i] = 2
    elif altman_z[i] > 1.8:
        y[i] = 1
    else:
        y[i] = 0

X = np.column_stack([X1, X2, X3, X4, X5])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train_scaled, y_train)

accuracy = model.score(X_test_scaled, y_test)
print(f"Tacnost modela: {accuracy:.2%}")

# Cuvanje modela
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model sacuvan! Sada restartuj aplikaciju.")