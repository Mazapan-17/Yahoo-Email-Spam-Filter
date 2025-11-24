import imaplib
from email import message_from_bytes

# ============================================================================
# CONFIGURATION - Fill in your details
# ============================================================================
YAHOO_EMAIL = "your_email@yahoo.com"
APP_PASSWORD = "your_app_password_here"
# ============================================================================

print("Testing Yahoo IMAP connection...")
print("="*70)

try:
    # Connect to Yahoo IMAP server
    print("Connecting to Yahoo IMAP server...")
    mail = imaplib.IMAP4_SSL('imap.mail.yahoo.com')

    # Login
    print(f"Logging in as {YAHOO_EMAIL}...")
    mail.login(YAHOO_EMAIL, APP_PASSWORD)
    print("✅ Login successful!")

    # Select inbox
    print("Selecting inbox...")
    status, messages = mail.select('INBOX')
    total_emails = int(messages[0])
    print(f"✅ Inbox selected. Total emails: {total_emails}")

    # Search for unread emails
    print("\nSearching for unread emails...")
    status, email_ids = mail.search(None, 'UNSEEN')

    unread_ids = email_ids[0].split()
    print(f"✅ Found {len(unread_ids)} unread emails")

    # Fetch just ONE email as a test
    if len(unread_ids) > 0:
        print("\n" + "="*70)
        print("Fetching first unread email as test...")
        print("="*70)

        first_id = unread_ids[0]
        status, msg_data = mail.fetch(first_id, '(RFC822)')

        # Parse the email
        raw_email = msg_data[0][1]
        email_message = message_from_bytes(raw_email)

        # Display basic info
        print(f"Email ID: {first_id.decode()}")
        print(f"From: {email_message.get('From', 'Unknown')}")
        print(f"Subject: {email_message.get('Subject', 'No Subject')}")
        print(f"Date: {email_message.get('Date', 'Unknown')}")

        print("\n✅ Successfully fetched and parsed email!")
    else:
        print("⚠️  No unread emails found to test with")
        print("   (You might have already read all your emails)")

    # Close connection
    mail.close()
    mail.logout()
    print("\n✅ Connection test complete!")
    print("="*70)
    print("Your IMAP connection is working! Ready to integrate with spam scanner.")

except imaplib.IMAP4.error as e:
    print(f"\n❌ IMAP Error: {e}")
    print("Possible issues:")
    print("  - Wrong email or app password")
    print("  - App password not generated correctly")
    print("  - IMAP not enabled in Yahoo settings")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("Check your internet connection and try again.")
