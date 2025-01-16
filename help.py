import pandas as pd

# Load the original dataset
df = pd.read_csv('preprocessed_data.csv') 
print(df.columns) # Replace with your actual dataset file name

# # Step 1: Print all column names to check if 'Gender' exists or if there's a typo
# print("Column names in the dataset:", df.columns)

# # Step 2: If the 'Gender' column exists, map 'M' to 0.0 and 'F' to 1.0
# if 'Gender' in df.columns:
#     df['Gender'] = df['Gender'].map({'M': 0.0, 'F': 1.0})
# else:
#     print("Error: 'Gender' column not found. Please check the column name.")

# # Step 3: Save the modified dataset to a new CSV file
# df.to_csv('modified_dataset.csv', index=False)  # Save the modified dataset to a new CSV file

# print("Dataset saved as 'modified_dataset.csv'.")
