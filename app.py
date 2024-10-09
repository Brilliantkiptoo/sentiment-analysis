import nltk

# Set NLTK data path
nltk.data.path.append('/home/vscode/nltk_data')

# Import other libraries
import streamlit as st
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize
import requests
from bs4 import BeautifulSoup
import re

# Download NLTK data
# nltk.download('punkt', quiet=True)

# Load the pre-trained models
@st.cache_resource
def load_models():
    doc2vec_model = Doc2Vec.load("doc2vec_model.model")
    lstm_model = load_model("lstm_model.h5")
    return doc2vec_model, lstm_model

doc2vec_model, lstm_model = load_models()

def preprocess_text(text):
    # Remove newlines and extra spaces
    text = re.sub('\s+', ' ', text)
    return text.strip()

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return preprocess_text(text)
    except:
        st.error("Error fetching content from the URL. Please check the URL and try again.")
        return None

def predict_sentiment(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Generate document vector
    doc_vector = doc2vec_model.infer_vector(tokens)
    
    # Reshape for LSTM input
    lstm_input = doc_vector.reshape(1, -1, 1)
    
    # Predict using LSTM model
    prediction = lstm_model.predict(lstm_input)
    
    # Get the predicted class
    predicted_class = np.argmax(prediction)
    
    # Map class to sentiment
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[predicted_class], prediction[0]

st.title("Financial News Sentiment Analysis")

input_type = st.radio("Choose input type:", ("Text", "URL"))

if input_type == "Text":
    user_input = st.text_area("Enter the financial news text:")
else:
    user_input = st.text_input("Enter the URL of the financial news article:")

if st.button("Analyze Sentiment"):
    if user_input:
        with st.spinner("Analyzing sentiment..."):
            if input_type == "URL":
                text = extract_text_from_url(user_input)
                if text is None:
                    st.stop()
            else:
                text = preprocess_text(user_input)
            
            sentiment, probabilities = predict_sentiment(text)
            
            st.subheader("Sentiment Analysis Result:")
            st.write(f"The sentiment of the text is: **{sentiment}**")
            
            st.subheader("Sentiment Probabilities:")
            st.bar_chart({"Negative": probabilities[0], "Neutral": probabilities[1], "Positive": probabilities[2]})
            
            if input_type == "URL":
                st.subheader("Extracted Text:")
                st.write(text[:1000] + "..." if len(text) > 1000 else text)
    else:
        st.warning("Please enter some text or a URL to analyze.")

