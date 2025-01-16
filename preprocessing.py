import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv("Theek.csv")  # Replace with your dataset path

# Handle 'Gender' mapping
df['Gender'] = df['Gender'].map({'M': 0.0, 'F': 1.0})  # Map 'M' to 0.0 and 'F' to 1.0

# Columns for preprocessing
numeric_features = [
    "Pain Level (0-10)", "Mobility Score", "Fatigue Level (1-5)",
    "Recovery Milestones Achieved", "Adherence to Recommendations (%)",
    "Daily Activity Levels", "Sleep Quality (1-10)", "Dietary Habits (1-5)",
    "Symptom Severity"  # Ensure Symptom Severity is part of numeric columns
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

# Preprocessing pipeline
preprocessed_data = preprocessor.fit_transform(df)

# Convert preprocessed data to DataFrame and retain column names
numeric_columns = numeric_features
categorical_columns = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
columns = numeric_columns + list(categorical_columns)

preprocessed_df = pd.DataFrame(preprocessed_data, columns=columns)

# Save preprocessed data
preprocessed_df.to_csv("preprocessed_data.csv", index=False)

print("Preprocessing complete. Preprocessed data saved to 'preprocessed_data.csv'.")
