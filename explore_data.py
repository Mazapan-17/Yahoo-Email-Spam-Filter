import pandas as pd

# Loads the CSV file
df = pd.read_csv('spam_assassin.csv')

print("Dataset shape (rows, columns):", df.shape)
print("\n" + "="*50)
print("Column names:")
print(df.columns.tolist())
print("\n" + "="*50)
print("First few rows:")
print(df.head())
print("\n" + "="*50)
print("Data types:")
print(df.dtypes)
