import os
import matplotlib.pyplot as plt
from email import policy
from email.parser import BytesParser

# ---------------- LOAD DATA ---------------- #

spam_emails_path = os.path.join("spamassassin-public-corpus", "spam")
ham_emails_path = os.path.join("spamassassin-public-corpus", "ham")

labeled_file_directories = [
    (spam_emails_path, 0),  # 0 = Spam
    (ham_emails_path, 1)    # 1 = Ham
]

email_corpus = []
labels = []

for class_files, label in labeled_file_directories:
    files = os.listdir(class_files)
    
    for file in files:
        file_path = os.path.join(class_files, file)

        try:
            with open(file_path, "r", encoding="latin-1", errors="ignore") as currentFile:
                email_content = currentFile.read().replace("\n", " ")
                email_corpus.append(email_content)
                labels.append(label)
        except:
            pass

# ---------------- HISTOGRAM ---------------- #

spam_count = labels.count(0)
print("Spam Mails = ",spam_count)
ham_count = labels.count(1)
print("Ham (Not Spam) Mails = ", ham_count)

plt.figure()
plt.bar(["Spam", "Ham"], [spam_count, ham_count])
plt.xlabel("Email Type")
plt.ylabel("Number of Emails")
plt.title("Spam vs Ham Email Distribution")
plt.show()

# ---------------- TRAIN TEST SPLIT ---------------- #

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    email_corpus, labels, test_size=0.2, random_state=11
)

# ---------------- PIPELINE ---------------- #

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn import tree

nlp_followed_by_dt = Pipeline([
    ("vect", HashingVectorizer(input="content", ngram_range=(1,3))),
    ("tfidf", TfidfTransformer(use_idf=True)),
    ("dt", tree.DecisionTreeClassifier(class_weight="balanced"))
])

nlp_followed_by_dt.fit(X_train, y_train)

# ---------------- MODEL EVALUATION ---------------- #

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_test_pred = nlp_followed_by_dt.predict(X_test)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=["Spam", "No-Spam"]))

# ---------------- .EML FILE PREDICTION ---------------- #

def extract_text_from_eml(file_path):
    with open(file_path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    body = ""

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_content()
    else:
        body = msg.get_content()

    return body


def predict_eml(file_path):
    email_text = extract_text_from_eml(file_path).replace("\n", " ")

    prediction = nlp_followed_by_dt.predict([email_text])[0]
    probability = nlp_followed_by_dt.predict_proba([email_text])[0]

    spam_prob = probability[0]
    ham_prob = probability[1]

    if prediction == 0:
        result = "SPAM"
    else:
        result = "NOT SPAM"

    print("\nPrediction Result:")
    print("Classification:", result)
    print(f"Spam Probability: {spam_prob:.4f}")
    print(f"Ham Probability:  {ham_prob:.4f}")


# ---------------- USER INPUT ---------------- #

file_path = input("Enter your .eml file path: ")
predict_eml(file_path)
