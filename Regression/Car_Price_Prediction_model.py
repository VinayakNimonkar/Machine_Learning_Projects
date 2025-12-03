# ---------------------------------------------------------
# 1. Import Libraries
# ---------------------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------
# 2. Load Dataset
# ---------------------------------------------------------
data = pd.read_csv(r"C:\Users\Phoenix\Downloads\archive (1)\test-data.csv")

# Remove missing values
data.dropna(inplace=True)

# ---------------------------------------------------------
# 3. Label Encoding
# ---------------------------------------------------------
enc_name = LabelEncoder()
enc_fuel = LabelEncoder()
enc_owner = LabelEncoder()
enc_trans = LabelEncoder()
enc_loc = LabelEncoder()

data['Name'] = enc_name.fit_transform(data['Name'])
data['Fuel_Type'] = enc_fuel.fit_transform(data['Fuel_Type'])
data['Owner_Type'] = enc_owner.fit_transform(data['Owner_Type'])
data['Transmission'] = enc_trans.fit_transform(data['Transmission'])
data['Location'] = enc_loc.fit_transform(data['Location'])

# ---------------------------------------------------------
# 4. Split Input / Output
# ---------------------------------------------------------
X = data.drop(columns=['selling_price'])
y = data['selling_price']

# ---------------------------------------------------------
# 5. Feature Scaling
# ---------------------------------------------------------
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ---------------------------------------------------------
# 6. Train-Test Split
# ---------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 7. Model Training
# ---------------------------------------------------------
model = RandomForestRegressor(n_estimators=120, random_state=42)
model.fit(x_train, y_train)

# ---------------------------------------------------------
# 8. Accuracy
# ---------------------------------------------------------
train_acc = model.score(x_train, y_train) * 100
test_acc = model.score(x_test, y_test) * 100

print("Training Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)
print("MAE:", mean_absolute_error(y_test, model.predict(x_test)))
print("MSE:", mean_squared_error(y_test, model.predict(x_test)))

# ---------------------------------------------------------
# 9. Prediction on New Data
# ---------------------------------------------------------
new_data = pd.DataFrame([[
    'Maruti Alto K10 LXI CNG',   # Name
    'Delhi',                     # Location
    2014,                        # Year
    40929,                       # KM Driven
    'CNG',                       # Fuel Type
    'Manual',                    # Transmission
    'First',                     # Owner Type
    12.5,                        # Mileage
    998,                         # Engine
    58.2,                        # Power
    4,                           # Seats
    5.59                         # Age?
]], columns=X.columns)

# Apply SAME encoders (only transform)
new_data['Name'] = enc_name.transform(new_data['Name'])
new_data['Fuel_Type'] = enc_fuel.transform(new_data['Fuel_Type'])
new_data['Transmission'] = enc_trans.transform(new_data['Transmission'])
new_data['Owner_Type'] = enc_owner.transform(new_data['Owner_Type'])
new_data['Location'] = enc_loc.transform(new_data['Location'])

# Scaling new data
new_data_scaled = pd.DataFrame(scaler.transform(new_data), columns=new_data.columns)

# Predict
predicted_price = model.predict(new_data_scaled)
print("Predicted Price:", predicted_price[0])
