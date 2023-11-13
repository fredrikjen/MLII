import streamlit as st
import pandas as pd
import joblib

# Load your trained model and the scaler
model = joblib.load("wine_fraud_final.pkl")
scaler = joblib.load("scaler.pkl")

# Function to predict wine fraud
def predict_wine_fraud(features):
    # Scale the features using the loaded scaler
    features_scaled = scaler.transform([features])
    # Predict using the SVM classifier
    prediction = model.predict(features_scaled)
    return prediction[0]

# Streamlit application layout
st.title('Wine Fraud Prediction')

# Input fields for the features your model requires
fixed_acidity = st.number_input("Fixed Acidity", format="%.2f")
volatile_acidity = st.number_input("Volatile Acidity", format="%.2f")
citric_acid = st.number_input("Citric Acid", format="%.2f")
residual_sugar = st.number_input("Residual Sugar", format="%.2f")
chlorides = st.number_input("Chlorides", format="%.2f")
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", format="%.2f")
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", format="%.2f")
density = st.number_input("Density", format="%.2f")
pH = st.number_input("pH", format="%.2f")
sulphates = st.number_input("Sulphates", format="%.2f")
alcohol = st.number_input("Alcohol", format="%.2f")
type_red = st.number_input("Type Red", format="%.2f")
type_white = st.number_input("Type White", format="%.2f")

# Button to make prediction
if st.button('Predict'):
    features = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
                free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, 
                alcohol, type_red, type_white]
    prediction = predict_wine_fraud(features)
    if prediction == 1:
        st.success("The wine is predicted to be Fraud")
    else:
        st.success("The wine is predicted to be Legit")
