Local AI Email Spam Filter
A machine learning project that classifies email messages as spam or legitimate (ham) using a locally-trained Naive Bayes classifier with TF-IDF feature extraction.
Project Goals
This project was built as a learning exercise to:
Understand the end-to-end machine learning workflow
Work with real email data and IMAP protocols
Train and evaluate a text classification model
Apply ML to solve a practical problem (inbox management)
Build a portfolio-worthy project demonstrating ML fundamentals
Results
Model Performance
Training Accuracy: 99.09%
Testing Accuracy: 98.45%
Spam Precision: 99.5% (when it says "spam", it's right 99.5% of the time)
Spam Recall: 95.8% (catches 95.8% of all spam emails)
Real-World Testing
Tested on 100+ real unread emails from a Yahoo inbox:
Successfully classified 97% as spam with minimal false positives
Properly decoded email headers and subjects
Identified legitimate emails with high accuracy
When tested on emails already in spam folder:
Correctly identified 86.3% as spam
Some missed patterns suggest room for customization with user-specific training data
Technical Stack
Python 3.x
scikit-learn - Machine learning (Naive Bayes, TF-IDF)
pandas - Data manipulation
imaplib - Email access via IMAP
email library - Email parsing

Project Structure
spam-filter-project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── explore_data.py              # Initial data exploration
├── inspect_labels.py            # Label analysis
├── preprocess_data.py           # Text cleaning and train/test split
├── train_model.py               # Model training and evaluation
├── scan_yahoo_improved.py       # Production scanner with whitelist
├── test_imap.py                 # IMAP connection testing
├── spam_classifier.pkl          # Trained model (not in repo)
├── vectorizer.pkl               # Trained TF-IDF vectorizer (not in repo)
└── data/
    ├── spam_assassin.csv        # Training data (not in repo)
    ├── train_data.csv           # Processed training set
    └── test_data.csv            # Processed test set
How It Works
1. Data Preparation
Used the SpamAssassin public corpus (5,796 emails: 67% ham, 33% spam)
Cleaned email text by removing headers, HTML, URLs, and special characters
Applied text normalization (lowercase, whitespace removal)
Split data into 80% training, 20% testing with stratification
2. Feature Extraction
TF-IDF (Term Frequency-Inverse Document Frequency) converts text to numerical features
Focuses on distinctive words that differentiate spam from legitimate email
Vocabulary size: 3,000 most important words and word pairs
3. Model Training
Naive Bayes classifier - ideal for text classification
Learns probability distributions of words in spam vs. ham
Fast training and prediction
Works well even with limited training data
4. Email Scanning
Connects to Yahoo Mail via IMAP
Properly decodes email headers (UTF-8, Base64 encoding)
Scans newest emails first
Supports whitelist for trusted senders
Outputs detailed CSV reports with spam probabilities

Key Learnings
What I Learned
Data preprocessing is critical - Raw email data requires extensive cleaning
Email encoding matters - Had to implement proper header decoding for non-ASCII characters
Class imbalance considerations - Stratified splitting preserves spam/ham ratios
Evaluation beyond accuracy - Precision and recall matter differently for spam filtering (false positives worse than false negatives)
Real-world testing reveals gaps - Public training data doesn't capture all spam patterns in personal inboxes
Iterative improvement - Started with basic classifier, added encoding fixes, whitelist support
Challenges Overcome
Encoded email subjects - Initially showed gibberish, fixed with proper header decoding
IMAP folder navigation - Different email providers use different folder names
False positive management - Balancing spam detection with avoiding blocking legitimate emails
Large unread email backlog - Optimized to scan newest emails first
Future Improvements
Option A: Improve Accuracy
Add user-specific training data from manually labeled emails
Implement ensemble methods (combine multiple classifiers)
Use more sophisticated models (SVM, Random Forest, or neural networks)
Add sender reputation scoring
Implement DKIM/SPF authentication checks
Option B: Build Automation
Automatically move spam to designated folder
Mark processed spam as read
Schedule daily/hourly scans
Email digest of detected spam
Web interface for managing whitelist and reviewing classifications
Option C: Advanced Features
Integrate local LLM (Qwen2) for uncertain classifications
Multi-class classification (spam, promotional, important, personal)
Phishing detection with URL analysis
Attachment safety scanning
Learning from user feedback (active learning)

Getting Started
Prerequisites
bash
pip install -r requirements.txt
1. Train the Model
bash
# Explore the dataset
python explore_data.py

# Preprocess and split data
python preprocess_data.py

# Train classifier
python train_model.py
2. Test IMAP Connection
bash
# Edit test_imap.py with your credentials
python test_imap.py
3. Scan Your Inbox
bash
# Edit scan_yahoo_improved.py with your credentials
# Configure whitelist, folder, and search criteria
python scan_yahoo_improved.py

Important Notes
Security
Never commit passwords or API keys to version control
Use app-specific passwords for email access (not your main password)
Consider using environment variables for credentials
Privacy
Model files and email data are excluded from repo (see .gitignore)
Be careful when sharing email content or sender information
Limitations
Trained on 2002 email data (SpamAssassin corpus) - spam patterns evolve
Best for personal use; production systems need more robust architecture
No real-time learning - requires retraining to adapt to new spam patterns

Resources
Datasets Used
SpamAssassin Public Corpus
Learning Resources
scikit-learn documentation: Text Classification
IMAP Protocol: RFC 3501
Email Parsing: Python email library
Contributing
This is a learning project, but suggestions and improvements are welcome! Feel free to:
Open issues for bugs or questions
Submit pull requests with enhancements
Share your own spam filtering approaches
License
This project is for educational purposes. Training data subject to SpamAssassin corpus license.
Author
[Sergio Estevez-Perez]
GitHub: [@Mazapan-17]
LinkedIn: [sergio-estevez-perez]

Built as part of my journey to learn machine learning and AI development.