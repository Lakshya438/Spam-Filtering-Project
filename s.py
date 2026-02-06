import os
import re
import textwrap
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

# Function to format email for display
def format_email_display(email_text, width=76):
    """Format email text for console display with proper line wrapping"""
    lines = email_text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            formatted_lines.append('')
            continue
        
        wrapped = textwrap.fill(line, width=width, 
                               break_long_words=False, 
                               break_on_hyphens=False)
        formatted_lines.append(wrapped)
    
    return '\n'.join(formatted_lines)

# Function to read email from file
def read_email_from_file(file_path):
    """Read email content from a file"""
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    return content
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, read as binary and decode with errors='ignore'
        with open(file_path, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
            return content
            
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

# Function to print a box with text
def print_box(text, width=80):
    """Print text in a box"""
    print("┌" + "─" * (width - 2) + "┐")
    print("│" + text.center(width - 2) + "│")
    print("└" + "─" * (width - 2) + "┘")

# Function to print section header
def print_section(title, width=80):
    """Print a section header"""
    print("\n" + "┌" + "─" * (width - 2) + "┐")
    print("│ " + title.ljust(width - 3) + "│")
    print("└" + "─" * (width - 2) + "┘")

# Load and prepare the data
spam_emails_path = os.path.join("spamassassin-public-corpus", "spam")
ham_emails_path = os.path.join("spamassassin-public-corpus", "ham")
labeled_file_directories = [(spam_emails_path, 1), (ham_emails_path, 0)]

email_corpus = []
labels = []

print("Loading emails from SpamAssassin corpus...")

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

print(f"\nLoaded {len(email_corpus)} emails")
print(f"  • Spam emails (label 1): {labels.count(1)}")
print(f"  • Ham emails (label 0): {labels.count(0)}")
print(f"  • Dataset balance: {labels.count(1)/len(labels)*100:.1f}% spam, {labels.count(0)/len(labels)*100:.1f}% ham")

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

print("\nTraining the model...\n")
nlp_classifier.fit(X_train, y_train)
print("Model training complete!\n")

# Evaluate on test set
y_test_pred = nlp_classifier.predict(X_test)
y_train_pred = nlp_classifier.predict(X_train)

print("\nMODEL EVALUATION")
print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy:     {accuracy_score(y_test, y_test_pred):.4f}")

if accuracy_score(y_train, y_train_pred) - accuracy_score(y_test, y_test_pred) > 0.1:
    print("\n⚠️  Warning: Possible overfitting detected (training accuracy >> test accuracy)")

print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# print("\nClassification Report:")
# print(classification_report(y_test, y_test_pred, target_names=["Spam", "Ham"]))

# User input prediction function
def predict_email(email_text):
    processed = preprocess_email(email_text)
    prediction = nlp_classifier.predict([processed])[0]
    probabilities = nlp_classifier.predict_proba([processed])[0]
    
    if prediction == 0:
        result = "SPAM"
        confidence = probabilities[0] * 100
        emoji = "🚫"
    else:
        result = "HAM (Not Spam)"
        confidence = probabilities[1] * 100
        emoji = "✅"
    
    return result, confidence, emoji

# Function to display prediction results
def display_prediction(email_text, file_name=None, email_counter=1):
    """Display email and its prediction"""
    
    # Display the email with proper formatting
    print("\n" + "="*80)
    if file_name:
        print_section(f"EMAIL #{email_counter} - File: {file_name}", 80)
    else:
        print_section(f"EMAIL #{email_counter}", 80)
    
    print("\n📧 EMAIL CONTENT:")
    print("┌" + "─" * 78 + "┐")
    
    # Format and display email with proper wrapping
    formatted_email = format_email_display(email_text, width=76)
    lines = formatted_email.split('\n')
    
    # Limit display to first 30 lines for very long emails
    max_display_lines = 30
    if len(lines) > max_display_lines:
        display_lines = lines[:max_display_lines]
        truncated = True
    else:
        display_lines = lines
        truncated = False
    
    for line in display_lines:
        print("│ " + line.ljust(76) + " │")
    
    if truncated:
        print("│ " + "... (email truncated for display)".ljust(76) + " │")
    
    print("└" + "─" * 78 + "┘")
    
    # Get prediction
    result, confidence, emoji = predict_email(email_text)
    
    # Display result with formatting
    print_section("PREDICTION RESULT", 80)
    
    print(f"\n{emoji}  Classification: {result}")
    print(f"📊 Confidence:     {confidence:.2f}%")
    print("\n" + "="*80)

# Interactive prediction loop
print("\n" + "="*80)
print_box("EMAIL SPAM DETECTOR", 80)
print("="*80)
print("\nOptions:")
print("  1. Type or paste email text directly")
print("  2. Drag and drop an email file (.eml, .txt, .msg)")
print("  3. Enter file path manually")
print("\nCommands:")
print("  • 'quit' - Exit the program")
print("  • 'clear' - Clear screen")
print("="*80)

email_counter = 0

def read_multiline_input():
    lines = []
    while True:
        line = input()
        line = input()
        if line.strip() == "":  # Empty line ends input
            break
        lines.append(line)
    return "\n".join(lines)

while True:
    print("\n" + "─" * 80)
    print("\n📥 Enter email text, file path, or drag-drop file here: ")
    # If user wants to type/paste email
    user_input = read_multiline_input()

    
    if user_input.lower() == 'quit':
        print("\n" + "="*80)
        print_box("Thank you for using Email Spam Detector!", 80)
        print("="*80)
        print(f"\n📊 Total emails analyzed in this session: {email_counter}\n")
        break
    
    if user_input.lower() == 'clear':
        os.system('clear' if os.name == 'posix' else 'cls')
        print("\n" + "="*80)
        print_box("EMAIL SPAM DETECTOR", 80)
        print("="*80)
        continue
    
    if user_input.strip() == "":
        print("\n⚠️  Please enter some text or a file path.")
        continue
    
    email_counter += 1
    
    # Remove quotes if file path is quoted (happens when drag-dropping on some systems)
    user_input = user_input.strip('"').strip("'")
    
    # Check if input is a file path
    if os.path.isfile(user_input):
        try:
            print(f"\n📂 Reading file: {os.path.basename(user_input)}")
            email_content = read_email_from_file(user_input)
            print("✓ File loaded successfully!")
            display_prediction(email_content, os.path.basename(user_input), email_counter)
        except Exception as e:
            print(f"\n❌ Error reading file: {str(e)}")
            email_counter -= 1
    else:
        # Treat as direct text input
        display_prediction(user_input, None, email_counter)

print("\nGoodbye! 👋")