import os
import re
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from imblearn.under_sampling import RandomUnderSampler
import email
from email import policy
from sklearn.metrics import confusion_matrix, accuracy_score

# ------------------- PREPROCESSING -------------------
def preprocess_email(email_text):
    try:
        msg = email.message_from_string(email_text, policy=policy.default)

        if msg.is_multipart():
            body = ""
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode("utf-8", errors="ignore")
        else:
            body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
    except:
        body = email_text

    body = re.sub(r"http\S+|www\S+", " URL ", body)
    body = re.sub(r"\S+@\S+", " EMAIL ", body)
    body = re.sub(r"\s+", " ", body)

    return body.strip().lower()

# ------------------- LOAD DATA -------------------
spam_path = "spamassassin-public-corpus/spam"
ham_path = "spamassassin-public-corpus/ham"

data = []
labels = []

for folder, label in [(spam_path, 1), (ham_path, 0)]:
    for file in os.listdir(folder):
        try:
            with open(os.path.join(folder, file), "r", encoding="latin-1") as f:
                content = preprocess_email(f.read())
                data.append(content)
                labels.append(label)
        except:
            pass

print("Total emails:", len(data))
print("Spam:", labels.count(1), "Ham:", labels.count(0))

# ------------------- SPLIT -------------------
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, stratify=labels, random_state=42
)

# ------------------- UNDERSAMPLE -------------------
rus = RandomUnderSampler(random_state=42)
X_train, y_train = rus.fit_resample(np.array(X_train).reshape(-1, 1), y_train)
X_train = X_train.flatten().tolist()

# ------------------- MODEL -------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=4000,
    stop_words="english",
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)

model = MultinomialNB(alpha=0.15)
model.fit(X_train_vec, y_train)

# ------------------- SAVE MODEL -------------------
with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")


# Vectorize test data
X_test_vec = vectorizer.transform(X_test)

# Predict on test data
y_pred = model.predict(X_test_vec)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)

print("Model Accuracy:", acc * 100, "%")

model.fit(X_train_vec, y_train)


# Transform test data using trained vectorizer
X_test_vec = vectorizer.transform(X_test)

# Predict spam/ham
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
print("✅ Accuracy on test set:", round(acc * 100, 2), "%")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


