# =============================================================
#  Fake Job Fraud Detection Model
#  Cleaned, Optimized & GitHub-Ready Version
# =============================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions
import html

# =============================================================
# 1. Load Dataset
# =============================================================
data = pd.read_csv(r"C:\Users\Phoenix\Downloads\archive (9)\fake_job_postings.csv")

# Remove salary_range (train model साठी irrelevant)
data.drop(columns=['salary_range'], inplace=True)

# =============================================================
# 2. Fill Missing Values
# =============================================================
for col in data.select_dtypes(include='object').columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# =============================================================
# 3. Label Encoding
# =============================================================
label_en = LabelEncoder()
cat_cols = ['title', 'location', 'department', 'company_profile', 'description',
            'requirements', 'benefits', 'employment_type', 'required_experience',
            'required_education', 'industry', 'function']

for col in cat_cols:
    data[col] = label_en.fit_transform(data[col])

# =============================================================
# 4. Train-Test Split
# =============================================================
X = data.drop(columns=['fraudulent'])
Y = data['fraudulent']

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# =============================================================
# 5. Train Models
# =============================================================

# Decision Tree Model
DT = DecisionTreeClassifier(max_depth=5)
DT.fit(x_train, y_train)

print("\nDecision Tree Accuracy:")
print("Test  :", DT.score(x_test, y_test) * 100)
print("Train :", DT.score(x_train, y_train) * 100)

# KNN Model
KNN = KNeighborsClassifier(n_neighbors=45)
KNN.fit(x_train, y_train)

print("\nKNN Accuracy:")
print("Test  :", KNN.score(x_test, y_test) * 100)
print("Train :", KNN.score(x_train, y_train) * 100)

# =============================================================
# 6. Decision Region Plot (Only 2 features allowed)
# =============================================================
X_plot = data[['has_company_logo', 'has_questions']]
y_plot = data['fraudulent']

scaler2 = StandardScaler()
X_plot = pd.DataFrame(scaler2.fit_transform(X_plot), columns=X_plot.columns)

KNN_plot = KNeighborsClassifier(n_neighbors=45)
KNN_plot.fit(X_plot, y_plot)

plot_decision_regions(X_plot.to_numpy(), y_plot.to_numpy(), clf=KNN_plot)
plt.xlabel("has_company_logo")
plt.ylabel("has_questions")
plt.title("Decision Region - Fake Job Prediction")
plt.show()

# =============================================================
# 7. Predict for NEW Input
# =============================================================

clean = lambda x: html.unescape(x).replace("Â", "").replace("â", "").strip()

new_data = pd.DataFrame([{
    'job_id': 999,
    'title': 'IC&E Technician',
    'location': 'US, Stockton, CA',
    'department': 'Oil & Energy',
    'company_profile': 'Staffing & recruiting for Oil & Energy industry.',
    'description': 'Calibrates, tests, maintains and troubleshoots control systems.',
    'requirements': 'High school diploma or GED. Four years experience as I&C Tech.',
    'benefits': clean("""
    Competitive pay, matched retirement fund, bonus structure,
    advancement opportunity, annual reviews, safe environment.
    """),
    'telecommuting': 0,
    'has_company_logo': 1,
    'has_questions': 1,
    'employment_type': 'Full-time',
    'required_experience': 'Mid-Senior level',
    'required_education': 'High School or equivalent',
    'industry': 'Oil & Energy',
    'function': 'Other'
}])

# Remove columns not present in training
new_data.drop(columns=['salary_range'], errors='ignore', inplace=True)

# Encode new data
for col in cat_cols:
    new_data[col] = label_en.fit_transform(new_data[col])

# Scale
scaled = scaler.transform(new_data)

# Predict
prediction = DT.predict(scaled)
print("\nPrediction for New Job Posting:", prediction[0])
