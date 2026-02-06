import streamlit as st
import pickle
import re
import email
from email import policy

# ------------------- LOAD MODEL -------------------
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ------------------- PREPROCESS -------------------
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

# ------------------- STREAMLIT UI -------------------
st.title("📧 Email Spam Detector")

st.write("Paste email text or upload file")

# TEXT BOX
email_text = st.text_area("Paste email content here:", height=200)

# FILE Uploader
uploaded_file = st.file_uploader("Or upload .txt or .eml file", type=["txt","eml"])

if uploaded_file:
    email_text = uploaded_file.read().decode("utf-8", errors="ignore")

# PREDICT BUTTON
if st.button("Check Spam"):
    if not email_text.strip():
        st.warning("Please enter or upload an email.")
    else:
        processed = preprocess_email(email_text)
        vectorized = vectorizer.transform([processed])
        
        probs = model.predict_proba(vectorized)[0]
        spam_prob = probs[1]
        ham_prob = probs[0]

        if spam_prob >= 0.4:
            st.error(f"🚨 SPAM DETECTED\nConfidence: {round(spam_prob*100,2)}%")
        else:
            st.success(f"✅ HAM EMAIL\nConfidence: {round(ham_prob*100,2)}%")

        st.markdown("### Processed Text:")
        st.code(processed[:500])
