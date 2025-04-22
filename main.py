import streamlit as st
import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
import os

# ---------- Text Cleaning ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# ---------- Load & Prepare Data ----------
@st.cache_data
def load_data():
    fake = pd.read_csv("C:/Users/panen/OneDrive/Desktop/fake news detection/Fake.csv", usecols=["text"])
    true = pd.read_csv("C:/Users/panen/OneDrive/Desktop/fake news detection/True.csv", usecols=["text"])
    fake["class"] = 0
    true["class"] = 1
    df = pd.concat([fake, true], ignore_index=True)
    df["text"] = df["text"].apply(clean_text)
    return df

# ---------- Train Model ----------
@st.cache_resource
def train_model(df):
    x = df["text"]
    y = df["class"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    vectorizer = TfidfVectorizer(max_features=10000)
    x_train_vec = vectorizer.fit_transform(x_train)
    model = DecisionTreeClassifier()
    model.fit(x_train_vec, y_train)
    return model, vectorizer

# ---------- Prediction ----------
def predict_news(news, model, vectorizer):
    cleaned = clean_text(news)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)[0]
    return "Fake" if result == 0 else "Real"

# ---------- Streamlit UI ----------
st.title("📰 Fake News Detection App")
st.write("Paste any news content below to analyze whether it's **Fake** or **Real**.")

news_input = st.text_area("✍️ Enter News Content Here", height=200)

if st.button("🔍 Analyze"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:
        with st.spinner("Analyzing..."):
            data = load_data()
            model, vectorizer = train_model(data)
            prediction = predict_news(news_input, model, vectorizer)

        if prediction == "Fake":
            st.markdown("<h2 style='color:red;'>🛑 Fake News Detected!</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color:green;'>✅ This is Real News!</h2>", unsafe_allow_html=True)
