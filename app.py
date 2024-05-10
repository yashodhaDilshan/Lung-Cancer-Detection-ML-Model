import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('LungCancerPredictorMF.pickle', 'rb') as file:
    model = pickle.load(file)

# Define the layout of the app
st.title('Lung Cancer Predictor')

# Add input widgets
age = st.slider('Age', 20, 90, 50)
smoking = st.radio('Smoking History', ['Yes', 'No'])
fatigue = st.radio('Fatigue', ['Yes', 'No'])
alcohol = st.radio('Alcohol Consuming', ['Yes', 'No'])
coughing = st.radio('Coughing', ['Yes', 'No'])
gender = st.radio('Gender', ['Male', 'Female'])
AreQ = st.radio('AreQ', ['Yes', 'No'])
Chronic_disease = st.radio('Chronic Disease', ['Yes', 'No'])
Anxiety = st.radio('Anxiety', ['Yes', 'No'])

# Preprocess user input and make prediction
smoking_encoded = 1 if smoking == 'Yes' else 0
fatigue_encoded = 1 if fatigue == 'Yes' else 0
alcohol_encoded = 1 if alcohol == 'Yes' else 0
coughing_encoded = 1 if coughing == 'Yes' else 0

# Make prediction
if gender == 'Male':
    gender_encoded = 1
    prediction = model.predict([[age, smoking_encoded, fatigue_encoded, alcohol_encoded, coughing_encoded, 1, 0]])
else:
    gender_encoded = 0
    prediction = model.predict([[age, smoking_encoded, fatigue_encoded, alcohol_encoded, coughing_encoded, 0, 1]])

# Display prediction result
if prediction == 0:
    st.write('Based on the input data, the model predicts that the individual does not have lung cancer.')
else:
    st.write('Based on the input data, the model predicts that the individual has lung cancer.')
