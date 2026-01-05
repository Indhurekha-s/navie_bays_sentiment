# streamlit_sentiment_app.py

import streamlit as st
import pickle

# load model
with open("sentiment_nb_model.pkl", "rb") as f:
    model = pickle.load(f)

# load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Step 2: App Title
st.title("Sentiment Analysis App")
st.write("Enter some text and see if the sentiment is Positive or Negative.")

# Step 3: User Input
user_input = st.text_area("Enter Text Here:")

# Step 4: Predict Button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        text_vec = vectorizer.transform([user_input])
        prediction = model.predict(text_vec)[0]
        probability = model.predict_proba(text_vec).max()

        # Step 5: Display Result
        st.subheader("Prediction Result:")
        st.write(f"Sentiment: **{prediction}**")
        st.write(f"Confidence: **{probability*100:.2f}%**")
