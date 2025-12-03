# ---------------------------------------------------------
# 1. Import Libraries
# ---------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
import numpy as np

# ---------------------------------------------------------
# 2. Load Dataset
# ---------------------------------------------------------
data = pd.read_csv(r"C:\Users\Phoenix\Downloads\archive (8)\Social_Network_Ads.csv")

print(data.head())
print("Total Null Values:", data.isnull().sum().sum())
print(data.describe())
print(data.info())

# ---------------------------------------------------------
# 3. Encode Gender
# ---------------------------------------------------------
encoder = OrdinalEncoder(categories=[['Male', 'Female']])
data['Gender'] = encoder.fit_transform(data[['Gender']])

# ---------------------------------------------------------
# 4. Input / Output Split
# ---------------------------------------------------------
X = data.drop(columns=['Purchased'])
y = data['Purchased']

# ---------------------------------------------------------
# 5. (Optional) Polynomial Features
# ---------------------------------------------------------
poly = PolynomialFeatures()
_ = poly.fit_transform(X)

# ---------------------------------------------------------
# 6. Scaling
# ---------------------------------------------------------
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ---------------------------------------------------------
# 7. Train-Test Split
# ---------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 8. Decision Tree Classifier
# ---------------------------------------------------------
model = DecisionTreeClassifier(max_depth=2)
model.fit(x_train, y_train)

# ---------------------------------------------------------
# 9. Accuracy
# ---------------------------------------------------------
print("Train Accuracy:", model.score(x_train, y_train) * 100)
print("Test Accuracy:", model.score(x_test, y_test) * 100)

# ---------------------------------------------------------
# 10. Check Different Depth Levels
# ---------------------------------------------------------
print("\nDepth-wise Performance:")
for depth in range(1, 21):
    temp = DecisionTreeClassifier(max_depth=depth)
    temp.fit(x_train, y_train)
    print(f"Depth {depth}: Train={temp.score(x_train, y_train)*100:.2f}% | Test={temp.score(x_test, y_test)*100:.2f}%")

# ---------------------------------------------------------
# 11. Predict New Data
# ---------------------------------------------------------
new_data = pd.DataFrame([{
    'User ID': 15694829,
    'Gender': 1.0,
    'Age': 32,
    'EstimatedSalary': 150000
}])

new_scaled = scaler.transform(new_data)
prediction = model.predict(new_scaled)
print("\nPredicted Purchased:", prediction[0])

# ---------------------------------------------------------
# 12. Decision Boundary Visualization (Age vs Salary)
# ---------------------------------------------------------
x_vis = X_scaled[['Age', 'EstimatedSalary']]
y_vis = y

model_vis = DecisionTreeClassifier(max_depth=2)
model_vis.fit(x_vis, y_vis)

plt.figure(figsize=(8, 6))
plot_decision_regions(x_vis.values, y_vis.values, clf=model_vis)
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title("Decision Boundary - Decision Tree")
plt.savefig("Social_Network_Purchased.jpg")
plt.show()
