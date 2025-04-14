import streamlit as st
import pickle
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer



st.title("Fake News Detection")

# Load the pre-trained vectorizer
with open("vector.pkl", "rb") as f:
    vectorization = pickle.load(f)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Custom CSS for fixed-width and multi-line text box
st.markdown(
    """
    <style>
    .fixed-width-textarea textarea {
        width: 400px !important;  /* Fixed width */
        height: 150px !important; /* Set a fixed height */
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Multi-line text input (textarea)
user_input = st.text_area(
    "Enter your text:",
    key="user_input",
    placeholder="Type here...",
    label_visibility="visible"
)

# Function to clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # Remove text inside square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    return text

# Add a "Predict" button
if st.button("Predict"):
    if user_input.strip():  # Ensure input is not empty
        user_input = clean_text(user_input)

        # Transform text using the trained vectorizer
        user_input_vector = vectorization.transform([user_input])

        # Predict using the trained model
        prediction = model.predict(user_input_vector)

        # Display result
        st.write("Prediction:", "🛑 **Fake News**" if prediction[0] == 1 else "✅ **Real News**")
    else:
        st.warning("⚠️ Please enter some text before predicting.")
