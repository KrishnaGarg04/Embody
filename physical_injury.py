# physical_injury_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
  # Import the preprocessed data from 'preprocessing.py'

# Define target and features
df = pd.read_csv("preprocessed_data.csv")
X = df.drop(columns=["Recovery Milestones Achieved"])  # Features
y = df["Recovery Milestones Achieved"]  # Target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error for Physical Injury Recovery Milestones prediction: {mse}")

# Optionally, save the trained model for later use
import joblib
joblib.dump(rf_model, 'physical_injury_model.pkl')
