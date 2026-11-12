import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier  # El equivalente nativo ultra-veloz de XGBoost/LightGBM en Scikit-Learn
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score

# 1. GENERACIÓN DE DATOS CON ALTA COMPLEJIDAD NO LINEAL
np.random.seed(42)
n_patients = 4000  # Más datos para que el árbol aprenda a la perfección

age = np.random.randint(30, 80, n_patients)
cholesterol = np.random.randint(150, 340, n_patients)
blood_pressure = np.random.randint(90, 180, n_patients)
max_heart_rate = np.random.randint(100, 210, n_patients)

scoring = (age * 0.25) + (cholesterol * 0.05) + (blood_pressure * 0.06) - (max_heart_rate * 0.03)
synergy = np.where((blood_pressure > 135) & (cholesterol > 230), 32, 0)
total_score = scoring + synergy
risk = np.where(total_score > np.percentile(total_score, 50), 1, 0)

df = pd.DataFrame({
    'Age': age,
    'Cholesterol': cholesterol,
    'Blood_Pressure': blood_pressure,
    'Max_Heart_Rate': max_heart_rate,
    'Risk': risk
})

print("==== KAGGLE-SPEED FRONTIER ARCHITECTURE ====")
print(f"Total Patient Records: {len(df)}")
print("-" * 60)

# 2. ENHANCED FEATURE ENGINEERING
df['Cardio_Synergy'] = (df['Blood_Pressure'] * df['Cholesterol']) / 100
df['Age_Risk_Ratio'] = df['Age'] / (df['Max_Heart_Rate'] + 1)
df['Pressure_Age_Interaction'] = df['Blood_Pressure'] * df['Age']

# 3. SPLIT AND TRANSFORM
X = df[['Age', 'Cholesterol', 'Blood_Pressure', 'Max_Heart_Rate', 'Cardio_Synergy', 'Age_Risk_Ratio', 'Pressure_Age_Interaction']]
y = df['Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. EXTREME GRADIENT BOOSTING OPTIMIZED MODEL (Velocidad C++ con máxima profundidad)
ultimate_model = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.15,
    max_depth=8,              # Árboles profundos para no perder ni una décima de precisión
    l2_regularization=0.1,    # Regularización ligera para mantener estabilidad
    early_stopping=True,       # Si ya llegó al 100%, se detiene para ahorrar milisegundos
    random_state=42
)

start_train = time.time()
ultimate_model.fit(X_train_scaled, y_train)
end_train = time.time()
training_time = end_train - start_train

# 5. INFERENCE SPEED & METRICS EVALUATION
start_inf = time.time()
y_pred = ultimate_model.predict(X_test_scaled)
end_inf = time.time()
inference_time = (end_inf - start_inf) * 1000

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n==== PERFECTED SPEED & ACCURACY RESULTS ====")
print(f"⏱️ Training Execution Time: {training_time:.4f} seconds (ULTRA FAST)")
print(f"⚡ Single-Batch Inference Latency: {inference_time:.2f} ms")
print(f"🚀 Model Production Accuracy: {accuracy * 100:.2f}%")
print(f"🩺 Clinical Recall (Sensitivity): {recall * 100:.2f}%")
print(f"🎯 F1-Score: {f1 * 100:.2f}%")
print("-" * 60)

# 6. FINAL CONFUSION MATRIX VIEW
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='inferno', xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'])
plt.title('Ultimate Optimized Model - Confusion Matrix')
plt.ylabel('Actual Clinical Diagnostics')
plt.xlabel('AI Predicted Risk')
plt.tight_layout()
plt.show()

from sklearn.model_selection import StratifiedKFold, cross_validate

print("==== STARTING PRODUCTION ROBUSTNESS VERIFICATION ====")
print("Executing 5-Fold Stratified Cross-Validation...")
print("-" * 60)

# 1. Definimos la estrategia de validación (5 pliegues manteniendo la proporción de Risk)
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. Definimos las métricas que queremos auditar en cada iteración
scoring_metrics = ['accuracy', 'recall', 'f1']

# 3. Ejecutamos la validación cruzada sobre todo el dataset (X_train_scaled, y_train)
cv_results = cross_validate(
    ultimate_model,
    X_train_scaled,
    y_train,
    cv=cv_strategy,
    scoring=scoring_metrics,
    n_jobs=-1
)

# 4. Extracción de métricas
cv_accuracies = cv_results['test_accuracy'] * 100
cv_recalls = cv_results['test_recall'] * 100
cv_f1s = cv_results['test_f1'] * 100

# 5. Desglose de resultados para el reporte técnico
print("\n📊 CROSS-VALIDATION BREAKDOWN PER FOLD:")
for i in range(5):
    print(f" Fold {i+1} -> Accuracy: {cv_accuracies[i]:.2f}% | Recall: {cv_recalls[i]:.2f}%")

print("-" * 60)
print("🏆 FINAL BENCHMARK VALIDATION SUMMARY:")
print(f" 🚀 Mean Cross-Validated Accuracy : {np.mean(cv_accuracies):.2f}% (+/- {np.std(cv_accuracies):.2f}%)")
print(f" 🩺 Mean Cross-Validated Recall   : {np.mean(cv_recalls):.2f}% (+/- {np.std(cv_recalls):.2f}%)")
print(f" 🎯 Mean Cross-Validated F1-Score : {np.mean(cv_f1s):.2f}%")
print("======================================================")

