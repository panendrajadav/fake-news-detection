# 📰 Fake vs Real News Detection

A machine learning-based web app to classify news articles as **Fake** or **Real**, using natural language processing and classification techniques. Built using Python, scikit-learn, and Streamlit for an interactive user interface.

---

## 🔍 Objective

To develop a lightweight and efficient tool that helps users verify the authenticity of news articles by analyzing their content in real-time.

---

## 🚀 Features

- Real-time news classification through a web interface
- Trained on 40,000+ labeled news articles (Fake and True)
- Custom text preprocessing (cleaning, lowercasing, removing noise)
- Feature extraction using **TF-IDF Vectorization**
- Classification using **Decision Tree Algorithm**
- Built using **Streamlit** for fast deployment and ease of use

---

## 🧠 How It Works

1. Loads and merges datasets from `Fake.csv` and `True.csv`
2. Cleans and preprocesses the text (removes punctuation, URLs, digits, etc.)
3. Converts text into numeric features using `TfidfVectorizer`
4. Trains a `DecisionTreeClassifier` on the processed data
5. Provides real-time predictions based on user input through a Streamlit app

---

## ⚙️ Technologies Used

- Python
- pandas
- scikit-learn
- Streamlit
- TfidfVectorizer
- re, string (for text cleaning)
- joblib (for saving/loading model and vectorizer)

---

## 🖥️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fake-news-detection.git
   cd fake-news-detection
