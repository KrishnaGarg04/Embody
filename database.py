from supabase import create_client, Client
import pandas as pd

# Supabase project credentials
SUPABASE_URL = "https://bhvmoiuwohfvoypddlli.supabase.co"  # Replace with your Supabase URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJodm1vaXV3b2hmdm95cGRkbGxpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzU4MjY4OTAsImV4cCI6MjA1MTQwMjg5MH0.g5dEtKn4qW0rh3OTaf5vH9mPHSQL8GTmFh1d-JaPJgs"  # Replace with your Supabase anon/public key

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load the dataset
dataset_path = "preprocessed_data.csv"  # Update with the correct dataset path
df = pd.read_csv(dataset_path)

# Rename columns to match Supabase schema (if needed)
df.columns = [col.lower().replace(" ", "_").replace("(", "").replace(")", "") for col in df.columns]

# Check for required columns
required_columns = [
    "pain_level_0-10", "mobility_score", "fatigue_level_1-5", 
    "recovery_milestones_achieved", "adherence_to_recommendations_", 
    "daily_activity_levels", "sleep_quality_1-10", "dietary_habits_1-5",
    "gender_0.0", "gender_1.0"
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"The following required columns are missing in the dataset: {missing_columns}")

# Insert data into Supabase
records = df.to_dict(orient="records")

try:
    print("Starting data insertion...")
    response = supabase.table("dataset").insert(records).execute()
    if response.get("error"):
        print(f"Error inserting records: {response['error']}")
    else:
        print(f"Successfully inserted {len(records)} records into Supabase.")
except Exception as e:
    print(f"An error occurred while inserting data: {e}")