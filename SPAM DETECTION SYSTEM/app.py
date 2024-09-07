import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import re

# Load the model and vectorizer
model = pickle.load(open('spam_classifier_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Function to clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

# Streamlit app
st.title("Spam Detection System")

# Input from user
user_input = st.text_area("Enter the message text:")

# Predict button
if st.button("Predict"):
    if user_input:
        # Clean and vectorize the user input
        user_input_clean = clean_text(user_input)
        user_input_vectorized = vectorizer.transform([user_input_clean])
        
        # Make the prediction
        prediction = model.predict(user_input_vectorized)
        result = 'Spam' if prediction[0] == 1 else 'Ham'
        
        # Display the result
        st.write(f"The message is classified as: **{result}**")
    else:
        st.write("Please enter a message to classify.")

