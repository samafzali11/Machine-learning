import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('diabetes_dataset_synthetic_100k.csv')
X = df.drop('diabetes', axis=1)
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = DecisionTreeClassifier(random_state=42, max_depth=None, min_samples_split=2)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f'Decision Tree Accuracy: {acc:.4f}')
print(f'Confusion Matrix:\n{cm}')

def predict_diabetes(input_dict):
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    print(f"Predicted: {'Diabetes' if pred==1 else 'No Diabetes'}")
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
