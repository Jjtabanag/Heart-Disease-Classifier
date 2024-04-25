import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load pre-trained model and necessary pre-processing steps
model = pickle.load(open('heart_disease_classifier.pkl', 'rb'))  # Load your trained heart disease classifier

# Input form for user to input feature values
st.title("Heart Disease Predictor")
st.text("""Welcome to the Heart Disease Predictor! 
This tool uses machine learning to assess the risk of heart disease based on your 
symptoms. Simply fill out the form below and click \"Predict\" to get your result.""")
# Create input fields for each feature
age = st.number_input("Age", min_value=0, max_value=150, value=50)
sex = st.radio("Sex", ["Male", "Female"])

chest_pain_type = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
resting_bp_s = st.number_input("Resting Blood Pressure (mmHg)", min_value=0, max_value=250, value=120)
cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
fasting_blood_sugar = st.radio("Fasting Blood Sugar (> 120 mg/dl)", ["Yes", "No"])
resting_ecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
max_heart_rate = st.number_input("Maximum Heart Rate", min_value=0, max_value=300, value=150)
exercise_angina = st.radio("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=10.0, value=0.0)
st_slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

# Preprocess input features
input_features = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'chest pain type': [chest_pain_type],
    'resting bp s': [resting_bp_s],
    'cholesterol': [cholesterol],
    'fasting blood sugar': [fasting_blood_sugar],
    'resting ecg': [resting_ecg],
    'max heart rate': [max_heart_rate],
    'exercise angina': [exercise_angina],
    'oldpeak': [oldpeak],
    'ST slope': [st_slope]
})

# Preporcess the input data
input_features['sex'] = input_features['sex'].apply(lambda x: 0 if x == 'Female' else 1)
input_features['chest pain type'] = input_features['chest pain type'].apply(lambda x: 1 if x == 'Typical Angina' else
                                                                                    (2 if x == 'Atypical Angina' else
                                                                                    (3 if x == 'Non-Anginal Pain' else
                                                                                    (4 if x == 'Asymptomatic' else None))))
input_features['fasting blood sugar'] = input_features['fasting blood sugar'].apply(lambda x: 1 if x == 'Yes' else 0)
input_features['resting ecg'] = input_features['resting ecg'].apply(lambda x: 0 if x == 'Normal' else
                                                                            (1 if x == 'ST-T wave abnormality' else
                                                                            (2 if x == 'Left ventricular hypertrophy' else None)))
input_features['exercise angina'] = input_features['exercise angina'].apply(lambda x: 1 if x == 'Yes' else 0)
input_features['ST slope'] = input_features['ST slope'].apply(lambda x: 1 if x == 'Upsloping' else
                                                                    (2 if x == 'Flat' else
                                                                    (3 if x == 'Downsloping' else None)))

print(input_features)

if st.button('Predict'):
    # Make prediction
    prediction = model.predict(input_features)
    
    # Display prediction
    if prediction == 1:
        st.header("Risk of Heart Disease: High")
    else:
        st.header("Risk of Heart Disease: Low")
