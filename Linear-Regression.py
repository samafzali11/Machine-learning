import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

df = pd.read_csv('diabetes.csv')
X = df.drop('diabetes', axis=1)
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred_cont = model.predict(X_test_scaled)
y_pred = (y_pred_cont > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred_cont)
print(f'Linear Regression Accuracy: {acc:.4f}')
print(f'Mean Squared Error: {mse:.4f}')

def predict_diabetes(input_dict):
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    pred_cont = model.predict(input_scaled)[0]
    pred = int(pred_cont > 0.5)
    prob = min(max(pred_cont,0),1)
    print(f"Predicted: {'Diabetes' if pred==1 else 'No Diabetes'} (probability {prob:.2f})")
    return pred

sample = {
    'age': 55,
    'bmi': 31,
    'blood_pressure': 95,
    'glucose': 150,
    'insulin': 90,
    'hba1c': 7.1,
    'cholesterol': 220,
    'hdl': 40,
    'ldl': 140,
    'triglycerides': 180,
    'family_history': 1,
    'physical_activity': 2,
    'diet_score': 4,
    'smoking': 0,
    'alcohol': 0,
    'gestational_diabetes': 0,
    'polyuria': 1,
    'polydipsia': 1,
    'neuropathy': 0,
    'retinopathy': 0,
    'weight_change': 3
}

predict_diabetes(sample)
