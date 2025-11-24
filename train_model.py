import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

print("Loading processed data...")
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")
print("="*50)

# Step 1: Convert text to numbers using TF-IDF
print("\nStep 1: Converting text to numbers (TF-IDF)...")
vectorizer = TfidfVectorizer(
    max_features=3000,  # Keep top 3000 most important words
    min_df=2,           # Word must appear in at least 2 emails
    max_df=0.8,         # Ignore words in more than 80% of emails
    ngram_range=(1, 2)  # Use single words and pairs of words
)

# Fit vectorizer on training data and transform
X_train = vectorizer.fit_transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])

print(f"Vocabulary size: {len(vectorizer.vocabulary_)} words")
print(f"Training matrix shape: {X_train.shape}")
print(f"Testing matrix shape: {X_test.shape}")

# Step 2: Train the classifier
print("\nStep 2: Training Naive Bayes classifier...")
classifier = MultinomialNB(alpha=0.1)  # alpha is smoothing parameter
classifier.fit(X_train, train_df['target'])
print("✅ Training complete!")

# Step 3: Make predictions
print("\nStep 3: Testing the model...")
y_pred_train = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)

# Step 4: Evaluate performance
print("\n" + "="*50)
print("PERFORMANCE RESULTS")
print("="*50)

train_accuracy = accuracy_score(train_df['target'], y_pred_train)
test_accuracy = accuracy_score(test_df['target'], y_pred_test)

print(f"\nTraining Accuracy: {train_accuracy*100:.2f}%")
print(f"Testing Accuracy: {test_accuracy*100:.2f}%")

print("\n" + "="*50)
print("DETAILED TEST RESULTS")
print("="*50)
print(classification_report(
    test_df['target'],
    y_pred_test,
    target_names=['Ham (Legitimate)', 'Spam'],
    digits=3
))

# Confusion Matrix
print("="*50)
print("CONFUSION MATRIX")
print("="*50)
cm = confusion_matrix(test_df['target'], y_pred_test)
print(f"True Ham (correctly identified): {cm[0][0]}")
print(f"False Spam (ham marked as spam): {cm[0][1]} ⚠️")
print(f"False Ham (spam marked as ham): {cm[1][0]} ⚠️")
print(f"True Spam (correctly identified): {cm[1][1]}")

# Step 5: Save the model
print("\n" + "="*50)
print("Saving model...")
with open('spam_classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✅ Model saved as: spam_classifier.pkl")
print("✅ Vectorizer saved as: vectorizer.pkl")

# Show some example predictions
print("\n" + "="*50)
print("EXAMPLE PREDICTIONS")
print("="*50)
for i in range(5):
    actual = "SPAM" if test_df['target'].iloc[i] == 1 else "HAM"
    predicted = "SPAM" if y_pred_test[i] == 1 else "HAM"
    correct = "✅" if actual == predicted else "❌"
    print(f"{correct} Actual: {actual:4s} | Predicted: {predicted:4s}")
    print(f"   Text: {test_df['text'].iloc[i][:100]}...")
    print()
