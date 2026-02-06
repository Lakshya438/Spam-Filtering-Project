import os
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import email
from email import policy

# Email preprocessing function
def preprocess_email(email_text):
    """Clean and preprocess email content"""
    try:
        # Parse email to get body content
        msg = email.message_from_string(email_text, policy=policy.default)
        
        # Extract body
        if msg.is_multipart():
            body = ""
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
        else:
            body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
    except:
        # If parsing fails, use raw text
        body = email_text
    
    # Remove URLs
    body = re.sub(r'http\S+|www\S+', ' URL ', body)
    
    # Remove email addresses
    body = re.sub(r'\S+@\S+', ' EMAIL ', body)
    
    # Remove excessive whitespace but keep some structure
    body = re.sub(r'\s+', ' ', body)
    
    return body.strip()

# Load and prepare the data
spam_emails_path = os.path.join("spamassassin-public-corpus", "spam")
ham_emails_path = os.path.join("spamassassin-public-corpus", "ham")
labeled_file_directories = [(spam_emails_path, 0), (ham_emails_path, 1)]

email_corpus = []
labels = []

print("Loading emails...")
for class_files, label in labeled_file_directories:    
    files = os.listdir(class_files)
    for file in files:
        file_path = os.path.join(class_files, file)
        try:
            with open(file_path, "r", encoding='latin-1') as currentFile:
                email_content = currentFile.read()
                processed_email = preprocess_email(email_content)
                email_corpus.append(processed_email)
                labels.append(label)
        except Exception as e:
            pass

print(f"Loaded {len(email_corpus)} emails")
print(f"Spam emails (label 0): {labels.count(0)}")
print(f"Ham emails (label 1): {labels.count(1)}")
print(f"Dataset balance: {labels.count(0)/len(labels)*100:.1f}% spam, {labels.count(1)/len(labels)*100:.1f}% ham")

# Split the data with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    email_corpus, labels, test_size=0.2, random_state=42, stratify=labels
)

# Create pipeline with better algorithm for text classification
nlp_classifier = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )),
    ("clf", MultinomialNB(alpha=0.1))
])

print("\nTraining the model...")
nlp_classifier.fit(X_train, y_train)

# Cross-validation to check for overfitting
print("\nPerforming cross-validation...")
cv_scores = cross_val_score(nlp_classifier, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Evaluate on test set
y_test_pred = nlp_classifier.predict(X_test)
y_train_pred = nlp_classifier.predict(X_train)

print("\n--- Model Evaluation ---")
print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

if accuracy_score(y_train, y_train_pred) - accuracy_score(y_test, y_test_pred) > 0.1:
    print("⚠️  Warning: Possible overfitting detected (training accuracy >> test accuracy)")

print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# print("\nClassification Report:")
# print(classification_report(y_test, y_test_pred, target_names=["Spam", "Ham"]))

# Function to display email preview
def display_email_preview(email_text, max_length=500):
    """Display a preview of the email content"""
    if len(email_text) > max_length:
        return email_text[:max_length] + "..."
    return email_text

# User input prediction function
def predict_email(email_text):
    processed = preprocess_email(email_text)
    prediction = nlp_classifier.predict([processed])[0]
    probabilities = nlp_classifier.predict_proba([processed])[0]
    
    if prediction == 0:
        result = "SPAM"
        confidence = probabilities[0] * 100
    else:
        result = "HAM (Not Spam)"
        confidence = probabilities[1] * 100
    
    return result, confidence

# Interactive prediction loop
print("\n" + "="*60)
print("EMAIL SPAM DETECTOR")
print("="*60)
print("Enter an email to check if it's spam or ham.")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("Enter email text (or 'quit' to exit): ")
    
    if user_input.lower() == 'quit':
        print("Exiting spam detector. Goodbye!")
        break
    
    if user_input.strip() == "":
        print("Please enter some text.\n")
        continue
    
    result, confidence = predict_email(user_input)
    print(f"\nPrediction: {result}")
    print(f"Confidence: {confidence:.2f}%\n")
    print("-" * 60 + "\n")