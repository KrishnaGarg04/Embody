from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

app = FastAPI()

# Load the trained models
mental_health_model = joblib.load('mental_health_model.pkl')
physical_injury_model = joblib.load('physical_injury_model.pkl')

# Define columns for preprocessing
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

class UserInput(BaseModel):
    pain_level: float
    mobility_score: float
    fatigue_level: float
    adherence: float
    daily_activity: float
    sleep_quality: float
    dietary_habits: float
    gender: str
    marital_status: int
    education_level: int
    household_gender: int

@app.post("/predict")
def predict(input: UserInput):
    # Convert the input to a DataFrame
    user_input = {
        "Pain Level (0-10)": [input.pain_level],
        "Mobility Score": [input.mobility_score],
        "Fatigue Level (1-5)": [input.fatigue_level],
        "Adherence to Recommendations (%)": [input.adherence * 100],  # Convert to percentage
        "Daily Activity Levels": [input.daily_activity],
        "Sleep Quality (1-10)": [input.sleep_quality],
        "Dietary Habits (1-5)": [input.dietary_habits],
        "Gender": [input.gender],
        "Marital Status": [input.marital_status],
        "Education Level (Adults 20+)": [input.education_level],
        "Household Reference Person Gender": [input.household_gender]
    }
    
    user_input_df = pd.DataFrame(user_input)
    
    # Preprocess the user input
    user_input_processed = preprocessor.fit_transform(user_input_df)
    
    # Extract feature names from the preprocessor
    processed_feature_names = preprocessor.get_feature_names_out()
    
    # Convert processed data into a DataFrame with feature names
    user_input_processed_df = pd.DataFrame(user_input_processed, columns=processed_feature_names)
    
    # Align input features with the model's expected features
    expected_features_mental_health = mental_health_model.feature_names_in_
    user_input_processed_df_mental = user_input_processed_df.reindex(columns=expected_features_mental_health, fill_value=0)
    
    expected_features_physical_injury = physical_injury_model.feature_names_in_
    user_input_processed_df_physical = user_input_processed_df.reindex(columns=expected_features_physical_injury, fill_value=0)
    
    # Make predictions
    mental_health_pred = mental_health_model.predict(user_input_processed_df_mental)
    physical_injury_pred = physical_injury_model.predict(user_input_processed_df_physical)
    
    return {
        "Predicted Mental Health Symptom Severity": mental_health_pred[0],
        "Predicted Physical Injury Recovery Milestones": physical_injury_pred[0]
    }
