import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

df = pd.read_csv('diabetes_dataset_synthetic_100k.csv')
X = df.drop('diabetes', axis=1)
y = df['diabetes']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=50, max_iter=500)
clusters = kmeans.fit_predict(X_scaled)

mapping = {}
for cluster in np.unique(clusters):
    indices = np.where(clusters == cluster)[0]
    majority_label = y.iloc[indices].mode()[0]
    mapping[cluster] = majority_label

y_pred_mapped = np.array([mapping[c] for c in clusters])
acc = accuracy_score(y, y_pred_mapped)
print(f'K-Means Clustering Accuracy (mapped to labels): {acc:.4f}')

def predict_diabetes(input_dict):
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    cluster = kmeans.predict(input_scaled)[0]
    pred = mapping[cluster]
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
