import pandas as pd
import pickle
import re
import imaplib
from email import message_from_bytes
from email.header import decode_header
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
YAHOO_EMAIL = "your_email@yahoo.com"
APP_PASSWORD = "your_app_password_here"
MAX_EMAILS_TO_SCAN = 100              # How many to scan

# Whitelist: Trusted senders that should NEVER be marked as spam
WHITELIST = [
    # Add trusted email addresses or domains here
    # Examples:
    # "friend@example.com",           # Specific email
    # "@company.com",                 # All emails from this domain
    # "@gmail.com",                   # Be careful with broad domains!
]

# What to scan: 'INBOX' or 'Spam' or '[Gmail]/Spam' (depends on your setup)
FOLDER_TO_SCAN = 'INBOX'  # Change to 'Spam' to test on spam folder
SEARCH_CRITERIA = 'ALL'  # 'UNSEEN' for unread, 'ALL' for everything
# ============================================================================

print("="*70)
print("IMPROVED YAHOO SPAM SCANNER")
print("="*70)

# Load trained model
print("\nLoading trained spam classifier...")
with open('spam_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
print("✅ Model loaded successfully!")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def decode_email_header(header):
    """Decode email headers that might be encoded"""
    if header is None:
        return "Unknown"

    decoded_parts = []
    for part, encoding in decode_header(header):
        if isinstance(part, bytes):
            try:
                decoded_parts.append(part.decode(
                    encoding or 'utf-8', errors='ignore'))
            except:
                decoded_parts.append(part.decode('utf-8', errors='ignore'))
        else:
            decoded_parts.append(str(part))

    return ''.join(decoded_parts)


def is_whitelisted(sender):
    """Check if sender is in whitelist"""
    sender_lower = sender.lower()
    for trusted in WHITELIST:
        trusted_lower = trusted.lower()
        if trusted_lower.startswith('@'):
            # Domain whitelist (e.g., @company.com)
            if trusted_lower in sender_lower:
                return True
        else:
            # Specific email whitelist
            if trusted_lower in sender_lower:
                return True
    return False


def clean_email(text):
    """Clean email text for classification"""
    if '\n\n' in text:
        text = text.split('\n\n', 1)[1]
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text


def extract_email_parts(email_message):
    """Extract and decode email components"""
    # Decode headers properly
    subject = decode_email_header(email_message.get('Subject'))
    sender = decode_email_header(email_message.get('From'))
    date = email_message.get('Date', 'Unknown')

    # Extract body
    body = ""
    if email_message.is_multipart():
        for part in email_message.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                try:
                    body = part.get_payload(decode=True).decode(
                        'utf-8', errors='ignore')
                    break
                except:
                    pass
    else:
        try:
            body = email_message.get_payload(
                decode=True).decode('utf-8', errors='ignore')
        except:
            body = str(email_message.get_payload())

    return subject, sender, date, body


def classify_email(subject, body, sender):
    """Classify email with whitelist support"""
    # Check whitelist first
    if is_whitelisted(sender):
        return False, 0.0, "whitelisted"

    # Combine subject and body
    full_text = f"{subject} {body}"

    # Clean and vectorize
    cleaned = clean_email(full_text)
    vectorized = vectorizer.transform([cleaned])

    # Predict
    prediction = classifier.predict(vectorized)[0]
    probability = classifier.predict_proba(vectorized)[0]

    is_spam = prediction == 1
    spam_probability = probability[1] * 100

    return is_spam, spam_probability, "ml_classified"

# ============================================================================
# CONNECT AND SCAN
# ============================================================================


print(f"\nConnecting to Yahoo IMAP...")
print(f"Scanning folder: {FOLDER_TO_SCAN}")
print(f"Search criteria: {SEARCH_CRITERIA}")

try:
    mail = imaplib.IMAP4_SSL('imap.mail.yahoo.com')
    mail.login(YAHOO_EMAIL, APP_PASSWORD)
    print("✅ Connected successfully!")

    # Select folder
    status, messages = mail.select(FOLDER_TO_SCAN)
    if status != 'OK':
        print(f"❌ Could not select folder: {FOLDER_TO_SCAN}")
        print("   Try 'INBOX', 'Spam', '[Gmail]/Spam', or 'Junk'")
        exit()

    # Search for emails
    status, email_ids = mail.search(None, SEARCH_CRITERIA)
    all_ids = email_ids[0].split()

    # REVERSE ORDER - newest first!
    all_ids = list(reversed(all_ids))

    total_found = len(all_ids)
    print(f"✅ Found {total_found} emails")

    # Limit scan
    emails_to_process = all_ids[:MAX_EMAILS_TO_SCAN]
    print(f"📧 Scanning {len(emails_to_process)} most recent emails...")
    print("="*70)

    # Process emails
    results = []
    spam_count = 0
    ham_count = 0
    whitelisted_count = 0

    for idx, email_id in enumerate(emails_to_process, 1):
        try:
            # Fetch email
            status, msg_data = mail.fetch(email_id, '(RFC822)')
            raw_email = msg_data[0][1]
            email_message = message_from_bytes(raw_email)

            # Extract and decode parts
            subject, sender, date, body = extract_email_parts(email_message)

            # Classify
            is_spam, spam_prob, classification_method = classify_email(
                subject, body, sender)

            # Store result
            results.append({
                'email_id': email_id.decode(),
                'date': date,
                'sender': sender[:60],
                'subject': subject[:100],
                'is_spam': is_spam,
                'spam_probability': round(spam_prob, 2),
                'method': classification_method
            })

            if classification_method == "whitelisted":
                whitelisted_count += 1
                ham_count += 1
            elif is_spam:
                spam_count += 1
            else:
                ham_count += 1

            # Progress
            if idx % 10 == 0:
                print(f"Processed {idx}/{len(emails_to_process)} emails...")

        except Exception as e:
            print(f"⚠️  Error processing email {email_id.decode()}: {e}")
            continue

    mail.close()
    mail.logout()
    print("✅ IMAP connection closed")

except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*70)
print("SCAN COMPLETE!")
print("="*70)
print(f"Total emails scanned: {len(results)}")
print(f"🟢 Legitimate (Ham): {ham_count} ({ham_count/len(results)*100:.1f}%)")
print(f"   └─ Whitelisted: {whitelisted_count}")
print(f"🔴 Spam detected: {spam_count} ({spam_count/len(results)*100:.1f}%)")

# Save results
if len(results) > 0:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('spam_probability', ascending=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = FOLDER_TO_SCAN.replace('/', '_')
    filename = f'scan_{folder_name}_{timestamp}.csv'
    results_df.to_csv(filename, index=False)
    print(f"\n✅ Results saved to: {filename}")

    # Show top spam
    print("\n" + "="*70)
    print("TOP 10 HIGHEST SPAM PROBABILITY:")
    print("="*70)
    top_spam = results_df.head(10)

    for idx, row in top_spam.iterrows():
        status_icon = "🔴" if row['is_spam'] else "🟢"
        method = " [WHITELIST]" if row['method'] == "whitelisted" else ""
        print(f"\n{status_icon} {row['spam_probability']:.1f}% spam{method}")
        print(f"   From: {row['sender']}")
        print(f"   Subject: {row['subject']}")

    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    print("1. Review the CSV file to check accuracy")
    print("2. Add trusted senders to WHITELIST in the script")
    print("3. To test on spam folder: change FOLDER_TO_SCAN = 'Spam'")
    print("4. To scan all emails: change SEARCH_CRITERIA = 'ALL'")
    print("="*70)
