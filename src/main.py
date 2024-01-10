import pandas as pd

# Load the dataset
dataset_path = 'data/UpdatedResumeDataSet.csv'
df = pd.read_csv(dataset_path)

# Display basic information about the dataset
print("Dataset Information:")
print(df.info())

# Display the first few rows of the dataset
print("\nSample Data:")
print(df.head())
