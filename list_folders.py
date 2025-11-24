import imaplib

# ============================================================================
# CONFIGURATION
# ============================================================================
YAHOO_EMAIL = "your_email@yahoo.com"
APP_PASSWORD = "your_app_password_here"
# ============================================================================

print("Connecting to Yahoo IMAP...")
try:
    mail = imaplib.IMAP4_SSL('imap.mail.yahoo.com')
    mail.login(YAHOO_EMAIL, APP_PASSWORD)
    print("✅ Connected successfully!\n")

    print("="*70)
    print("ALL AVAILABLE FOLDERS:")
    print("="*70)

    # List all folders - different approach
    status, folders = mail.list()

    if status == 'OK':
        folder_names = []
        for folder in folders:
            # Parse the folder line
            folder_str = folder.decode('utf-8')
            # Extract just the folder name (last part after quotes)
            parts = folder_str.split('"')
            if len(parts) >= 3:
                folder_name = parts[-2]
                folder_names.append(folder_name)
                print(f"  📁 {folder_name}")
            else:
                print(f"  📁 {folder_str}")

        print("\n" + "="*70)
        print("FOLDER NAMES TO TRY IN YOUR SCRIPT:")
        print("="*70)

        # Look for spam-related folders
        spam_keywords = ['spam', 'bulk', 'junk', 'trash']
        print("\nPotential spam folders:")
        for name in folder_names:
            if any(keyword in name.lower() for keyword in spam_keywords):
                print(f"  → FOLDER_TO_SCAN = '{name}'")

        print("\nAll folders (copy one of these):")
        for name in folder_names:
            print(f"  '{name}'")

    mail.logout()

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
