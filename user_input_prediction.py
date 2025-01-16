import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the trained models
mental_health_model = joblib.load('mental_health_model.pkl')  # Path to your mental health model
physical_injury_model = joblib.load('physical_injury_model.pkl')  # Path to your physical injury model

# Columns for preprocessing
numeric_features = [
    "Pain Level (0-10)", "Mobility Score", "Fatigue Level (1-5)",
    "Adherence to Recommendations (%)", "Daily Activity Levels", 
    "Sleep Quality (1-10)", "Dietary Habits (1-5)"
]
categorical_features = [
    "Gender", "Marital Status", "Education Level (Adults 20+)",
    "Household Reference Person Gender"
]

# Numeric preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing pipeline
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Sample user input (replace with actual input)
user_input = {
    "Pain Level (0-10)": [7],
    "Mobility Score": [3],
    "Fatigue Level (1-5)": [4],
    "Adherence to Recommendations (%)": [80],
    "Daily Activity Levels": [6],
    "Sleep Quality (1-10)": [5],
    "Dietary Habits (1-5)": [3],
    "Gender": ["M"],  # or 'F'
    "Marital Status": [1],
    "Education Level (Adults 20+)": [3],
    "Household Reference Person Gender": [2]
}

# Convert the input dictionary to a DataFrame
user_input_df = pd.DataFrame(user_input)

# Preprocess the user input
user_input_processed = preprocessor.fit_transform(user_input_df)

# Extract feature names from the preprocessor
processed_feature_names = preprocessor.get_feature_names_out()

# Convert processed data into a DataFrame with feature names
user_input_processed_df = pd.DataFrame(user_input_processed, columns=processed_feature_names)

# Align input features with the model's expected features
# For Mental Health model, ensure "Symptom Severity" is excluded from input, as it is the target.
# For Physical Injury model, ensure "Recovery Milestones Achieved" is excluded from input, as it is the target.

# Get expected features for mental_health_model
expected_features_mental_health = mental_health_model.feature_names_in_
user_input_processed_df = user_input_processed_df.reindex(columns=expected_features_mental_health, fill_value=0)

# Make predictions for Mental Health and Physical Injury using the respective models
mental_health_pred = mental_health_model.predict(user_input_processed_df)

# For physical injury model, also remove the "Recovery Milestones Achieved" column (target)
expected_features_physical_injury = physical_injury_model.feature_names_in_
user_input_processed_df = user_input_processed_df.drop(columns=["Recovery Milestones Achieved"], errors="ignore")
user_input_processed_df = user_input_processed_df.reindex(columns=expected_features_physical_injury, fill_value=0)

physical_injury_pred = physical_injury_model.predict(user_input_processed_df)

# Output the predictions
print(f"Predicted Mental Health Symptom Severity: {mental_health_pred[0]}")
print(f"Predicted Physical Injury Recovery Milestones: {physical_injury_pred[0]}")
