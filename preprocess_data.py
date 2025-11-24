import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('spam_assassin.csv')

print(f"Total Emails: {len(df)}")
print(f"Ham (legitimate): {len(df[df['target'] ==0])}")
print(f"Spam: {len(df[df['target'] ==1])}")
print("="*50)

# Function to clean email text


def clean_email(text):
    """
    Clean email text by removing headers and extracting meaningful content
    """

    # Remove email headers (everything before first blank line)
    # Headers end at first occurance of two newlines

    if '\n\n' in text:
        text = text.split('\n\n', 1)[1]     # Take everything after headers

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove special characters and digits (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Conver to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


# Apply cleaning
print("Clearning emails... this may take a minute")
df['cleaned_text'] = df['text'].apply(clean_email)

# Check a before/after example
print("\n" + "="*50)
print("BEFORE cleaning (first 300 chars):")
print(df['text'].iloc[1000][:300])
print("\n" + "="*50)
print("AFTER cleaning (first 300 chars):")
print(df['cleaned_text'].iloc[1000][:300])
print("\n" + "="*50)

# Split into training and testing sets
# 80% for training, 20% for testing

X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'],
    df['target'],
    test_size=0.2,
    random_state=42,
    stratify=df['target']   # Keep sam spam/ham ratio in both sets
)

print(f"\nTraining Set: {len(X_train)} emails")
print(f"\nTesting Set: {len(X_test)} emails")
print(f"\nTrain spam ratio: {(y_train==1).sum() / len(y_train) * 100:.1f}%")
print(f"\nTest spam ratio: {(y_test==1).sum() / len(y_test) * 100:.1f}%")

# Save processed data for next step
train_df = pd.DataFrame({'text': X_train, 'target': y_train})
test_df = pd.DataFrame({'text': X_test, 'target': y_test})

train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

print("\n Preprocessing complete!")
print("Created: train_data.csv and test_data.csv")
