import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load your trained model and label encoders
with open('gate_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder_gate.pkl', 'rb') as f:
    label_encoder_gate = pickle.load(f)

with open('label_encoder_park.pkl', 'rb') as f:
    label_encoder_park = pickle.load(f)

# Title of the app
st.title("Best Gate Prediction for Safari Sightings")

# User inputs
date_input = st.date_input("Select Date for Safari")
park_options = label_encoder_park.classes_  # Get the unique park names
selected_park = st.selectbox("Select National Park", park_options)

# Button to predict best gates
if st.button("Predict Best Gates"):
    # Prepare input data for prediction
    month = date_input.month
    day = date_input.day
    day_of_week = date_input.weekday()  # Monday=0, Sunday=6
    encoded_park = label_encoder_park.transform([selected_park])[0]

    input_data = [[month, day, day_of_week, encoded_park]]
    
    # Predict probabilities for each gate
    predicted_probs = model.predict_proba(input_data)[0]
    
    # Get the top 3 gates with the highest probabilities
    top_3_indices = np.argsort(predicted_probs)[-3:][::-1]
    top_3_gates = label_encoder_gate.inverse_transform(top_3_indices)

    # Display the results
    st.write("Best 3 Gates for sightings:")
    for gate in top_3_gates:
        st.write(f"- {gate}")

# Run the Streamlit app
# Save and run the following command in your terminal:
# streamlit run app.py
