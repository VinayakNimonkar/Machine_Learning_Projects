# ------------------------------------------------------------
# STUDENT SUCCESS PREDICTION - LOGISTIC REGRESSION MODEL
# ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

load_data = pd.read_csv(r"C:\Users\Phoenix\Downloads\student_success_datasetr.csv")

print("Sample Rows:")
print(load_data.head())

print("\nDataset Shape:")
print(f'Rows: {load_data.shape[0]}, Columns: {load_data.shape[1]}')

print("\nDataset Info:")
print(load_data.info())

print("\nStatistical Summary:")
print(load_data.describe(include='all'))

print("\nMissing Values:")
print(load_data.isnull().sum())

# ------------------------------------------------------------
# 2. LABEL ENCODING
# ------------------------------------------------------------

le = LabelEncoder()
load_data['internet'] = le.fit_transform(load_data['internet'])
load_data['passed'] = le.fit_transform(load_data['passed'])

print("\nAfter Encoding:")
print(load_data.head())
print(load_data.info())

# ------------------------------------------------------------
# 3. FEATURE SCALING
# ------------------------------------------------------------

features = ['studyhourse', 'attendance ', 'pass_score', 'internet', 'sleep_hourse']

scaler = StandardScaler()
scaled_data = load_data.copy()
scaled_data[features] = scaler.fit_transform(load_data[features])

X = scaled_data[features]
y = scaled_data['passed']

# ------------------------------------------------------------
# 4. TRAIN-TEST SPLIT
# ------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# 5. MODEL TRAINING
# ------------------------------------------------------------

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# ------------------------------------------------------------
# 6. MODEL EVALUATION
# ------------------------------------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Fail', 'Pass'],
            yticklabels=['Fail', 'Pass'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7. USER PREDICTION
# ------------------------------------------------------------

print("\n------ Predict Your Result ------")

try:
    studyhourse = float(input("Enter study hours: "))
    attendance = float(input("Enter attendance: "))
    pass_score = float(input("Enter past score: "))
    internet = int(input("Internet access? (1 = Yes, 0 = No): "))
    sleep_hourse = float(input("Enter sleep hours: "))

    user_input = pd.DataFrame([{
        'studyhourse': studyhourse,
        'attendance ': attendance,
        'pass_score': pass_score,
        'internet': internet,
        'sleep_hourse': sleep_hourse
    }])

    user_scaled = scaler.transform(user_input)

    prediction = model.predict(user_scaled)[0]
    result = "PASS ✅" if prediction == 1 else "FAIL ❌"

    print(f"\nPrediction: {result}")

except Exception as e:
    print("An error occurred:", e)
