import pandas as pd
import numpy as np
import joblib
import streamlit as st
from gmail_fetch import get_service, fetch_emails
import matplotlib.pyplot as plt


model = joblib.load('spam_model.pkl')

st.title("📧 Spam Email Classifier with Gmail")

if st.button("📩 Fetch My Gmail"):
    service = get_service()
    emails = fetch_emails(service)

from gmail_fetch import get_service, fetch_emails

service = get_service()                 # This triggers the login (which you completed)
emails = fetch_emails(service)
st.write("Fetched Emails:")
for i, email in enumerate(emails, 1):
    st.write(f"{i}. {email}")
# This actually fetches your recent emails


for i, text in enumerate(emails, 1):
        pred = model.predict([text])[0]
        proba = model.predict_proba([text])[0]
        label = "🚨 Spam" if pred == 1 else "✅ Ham"
        st.subheader(f"Email {i}: {label}")
        st.write(text)
        st.progress(proba[pred])
        fig, ax = plt.subplots()
        ax.pie(proba, labels=['Ham ✅', 'Spam 🚨'], autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
        ax.axis('equal')
        st.pyplot(fig)

