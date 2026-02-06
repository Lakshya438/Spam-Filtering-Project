import os
spam_emails_path = os.path.join("spamassassin-public-corpus",
"spam")
ham_emails_path = os.path.join("spamassassin-public-corpus", "ham")
labeled_file_directories = [(spam_emails_path, 0),
(ham_emails_path, 1)]

email_corpus = []
labels = []
for class_files, label in labeled_file_directories:    
    files = os.listdir(class_files)
    for file in files:
        file_path = os.path.join(class_files, file)
        try:
            with open(file_path, "r") as currentFile:
                email_content = currentFile.read().replace("\n","")
                email_content = str(email_content)
                email_corpus.append(email_content)
                labels.append(label)
                
        except:
            pass

print(email_content)
print("\n")
print(email_corpus)
print(label)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
email_corpus, labels, test_size=0.2, random_state=11
)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer,TfidfTransformer
from sklearn import tree
nlp_followed_by_dt = Pipeline(
[
("vect", HashingVectorizer(input="content", ngram_range=(1,
3))),("tfidf", TfidfTransformer(use_idf=True,)),
("dt",
tree.DecisionTreeClassifier(class_weight="balanced")),
]
)
nlp_followed_by_dt.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
y_test_pred = nlp_followed_by_dt.predict(X_test)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=["Spam", "No-Spam"]))


