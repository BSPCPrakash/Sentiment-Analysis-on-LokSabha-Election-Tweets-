import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load the trained model and vectorizer
clf = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocessing function to remove stopwords and tokenize
def preprocess(tweet):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(str(tweet))  # Ensure tweet is a string
    words = [word.lower() for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)  # Return the cleaned text as a string

# Streamlit app UI
st.title("Tweet Sentiment Analysis")
st.write("Enter a tweet to predict whether the sentiment is positive or negative.")

# Input from the user
tweet_input = st.text_area("Enter Tweet", "")

# Button to make a prediction
if st.button("Predict Sentiment"):
    if tweet_input:
        cleaned_tweet = preprocess(str(tweet_input))  # Preprocess the tweet
        tweet_vector = vectorizer.transform([cleaned_tweet])  # Use the same vectorizer
        sentiment = clf.predict(tweet_vector)[0]  # Predict sentiment

        # Display the result
        st.write(f"Sentiment of the tweet: **{sentiment}**")
    else:
        st.write("Please enter a tweet to predict its sentiment.")

