# mental_health_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load preprocessed dataset
df = pd.read_csv("preprocessed_data.csv")

# Define target and features
X = df.drop(columns=["Symptom Severity"])  # Features
y = df["Symptom Severity"]  # Target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest Regressor
rf_model_mental = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_mental.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model_mental.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error for Mental Health Symptom Severity prediction: {mse}")

# Save the trained model
joblib.dump(rf_model_mental, 'mental_health_model.pkl')

