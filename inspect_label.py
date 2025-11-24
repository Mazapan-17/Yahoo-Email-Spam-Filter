import pandas as pd

# Load the data
df = pd.read_csv('spam_assassin.csv')

print("Label distribution:")
print(df['target'].value_counts())
print("\n" + "="*50)

# Calculate percentages
print("Percentage breakdown:")
print(df['target'].value_counts())
print("\n" + "="*50)

# Look at a couple examples of each
print("Exmaple of target=0:")
print(df[df['target'] == 0]['text'].iloc[0][:500])  # First 500 Characters
print("\n" + "="*50)
print("Example of target=1:")
print(df[df['target'] == 0]['text'].iloc[0][:500])  # First 500 Characters
