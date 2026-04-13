import streamlit as st
import joblib
import numpy as np

# Load model + vectorizer
model = joblib.load("models/tuned_linear_svc.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")  # save this during training

st.set_page_config(page_title="Music Tag Predictor")

st.title(" Music Tagging System")
st.markdown("Predict tags from music captions")

# Input
caption = st.text_area("Enter music caption:")

if st.button("Predict Tags"):

    if caption.strip() == "":
        st.warning("Please enter a caption")
    else:
        # Transform text
        X = vectorizer.transform([caption])

        # Predict
        pred = model.predict(X)[0]

        # Get label indices
        predicted_labels = np.where(pred == 1)[0]

        if len(predicted_labels) == 0:
            st.write("No tags predicted")
        else:
            st.subheader("Predicted Tags")
            st.write(predicted_labels.tolist())
